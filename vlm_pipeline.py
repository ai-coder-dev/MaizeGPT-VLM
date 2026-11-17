"""
Maize VLM diagnosis pipeline (v2): SigLIP head + plant-part check + second-check VLM.

    farmer photo
      -> SigLIP (base, 384px) features, test-time augmentation (orig / flip / centre crop)
      -> calibrated classifier head over the trained IIMR problems
      -> SigLIP zero-shot plant-part check (leaf / stalk / cob / tassel / insect / not maize)
      -> second check by a vision LLM constrained to the full IIMR problem list, when the head
         is unsure, the photo is not a leaf, the class is confusable or under-trained
      -> fused top-3 causes + status:
           confident  -> diagnosis + ICAR-IIMR management (chemicals allowed)
           likely     -> top-3 with signs to check, scouting advice, KVK referral (no spray)
           uncertain  -> ask for a clearer photo of the affected part, show possible causes
           retake     -> not maize / unusable photo

Safety rules for the ICAR-IIMR demonstration:
* never say "healthy" unless the head is very sure AND the second check agrees;
* never give spray advice unless the status is "confident";
* chemicals come only from knowledge/advice.json, exported from the MaizeGPT table
  that is grounded in ICAR-IIMR publications (training/export_advice.py).
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent
HEAD_DIR = ROOT / "models" / "iimr_head_v2"
PROBLEMS_PATH = ROOT / "knowledge" / "iimr_problems.json"
ADVICE_PATH = ROOT / "knowledge" / "advice.json"
VISION_ID = "google/siglip-base-patch16-384"
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENT = 0.80          # SigLIP probability for a confident call without a second check
LIKELY = 0.45             # fused score below this is "uncertain"
HEALTHY_MIN = 0.90        # SigLIP probability needed before "healthy" is even considered
LOW_SUPPORT = 30          # classes trained on fewer images always get a second check
CONFUSABLE = {"charcoal_rot", "fusarium_stalk_rot", "bacterial_stalk_rot", "maydis_leaf_blight",
              "turcicum_leaf_blight", "grey_leaf_spot", "healthy", "zinc_deficiency", "thrips",
              "nitrogen_deficiency", "potassium_deficiency"}
LIKELIHOOD = {"high": 0.75, "medium": 0.45, "low": 0.15}

PART_PROMPTS = {
    "leaf": "a close-up photo of a maize leaf",
    "stalk": "a photo of a maize stalk or stem",
    "cob": "a photo of a maize cob or ear",
    "tassel": "a photo of a maize tassel",
    "whole_plant": "a photo of maize plants in a field",
    "insect": "a close-up photo of an insect, caterpillar or eggs",
    "not_maize": "a photo of something that is not a maize plant",
}

TIERS = {"pest": "critical", "disease": "moderate", "nutrient": "nutritional", "healthy": "healthy"}


@dataclass
class Problem:
    id: str
    name: str
    category: str
    parts: List[str]
    key_signs: str
    look_alikes: List[str]


def load_problems() -> Dict[str, Problem]:
    data = json.loads(PROBLEMS_PATH.read_text())["problems"]
    return {p["id"]: Problem(p["id"], p["name"], p["category"], p["parts"], p["key_signs"], p["look_alikes"])
            for p in data}


def load_advice() -> Dict[str, Dict]:
    if ADVICE_PATH.exists():
        return json.loads(ADVICE_PATH.read_text())
    return {}


class MaizeDiagnoser:
    def __init__(self, openai_api_key: Optional[str] = None, second_check_model: Optional[str] = None,
                 head_dir: Path = HEAD_DIR, enable_second_check: Optional[bool] = None):
        self.problems = load_problems()
        self.advice = load_advice()
        meta = json.loads((head_dir / "meta.json").read_text())
        self.classes: List[str] = meta["classes"]
        self.temperature: float = float(meta.get("temperature", 1.0))
        self.support: Dict[str, int] = meta.get("support", {})
        self.clf = joblib.load(head_dir / "classifier.joblib")
        self.scaler = joblib.load(head_dir / "scaler.joblib")

        self.processor = AutoProcessor.from_pretrained(VISION_ID)
        self.model = AutoModel.from_pretrained(VISION_ID, dtype=torch.float32, low_cpu_mem_usage=True).to(device).eval()
        with torch.no_grad():
            text_inputs = self.processor(text=list(PART_PROMPTS.values()), padding="max_length",
                                         return_tensors="pt").to(device)
            text = self.model.get_text_features(**text_inputs)
            if not isinstance(text, torch.Tensor):  # transformers >= 5 returns an output object
                text = text.pooler_output
            self.part_text = text / text.norm(dim=-1, keepdim=True)

        if enable_second_check is None:
            enable_second_check = os.getenv("VLM_SECOND_CHECK", "1") == "1"
        self.openai = None
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key and enable_second_check:
            from openai import OpenAI

            self.openai = OpenAI(api_key=api_key)
        self.second_check_model = second_check_model or os.getenv("SECOND_CHECK_MODEL", "gpt-6-luna")

    # ------------------------------------------------------------------ SigLIP
    @torch.no_grad()
    def siglip(self, image: Image.Image):
        """Calibrated class probabilities (TTA-averaged) and plant-part probabilities."""
        w, h = image.size
        crop = image.crop((int(w * 0.075), int(h * 0.075), int(w * 0.925), int(h * 0.925)))
        views = [image, ImageOps.mirror(image), crop]
        inputs = self.processor(images=views, return_tensors="pt").to(device)
        out = self.model.vision_model(inputs["pixel_values"])
        tokens = out.last_hidden_state
        feat = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        logits = self.clf.decision_function(self.scaler.transform(feat.cpu().numpy())) / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = (probs / probs.sum(axis=1, keepdims=True)).mean(axis=0)

        pooled = out.pooler_output[:1]
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        part_logits = (pooled @ self.part_text.T) * self.model.logit_scale.exp() + self.model.logit_bias
        part_probs = torch.softmax(part_logits[0], dim=-1).cpu().numpy()
        parts = dict(zip(PART_PROMPTS.keys(), part_probs.tolist()))
        return {c: float(p) for c, p in zip(self.classes, probs)}, parts

    # ------------------------------------------------------------ second check
    def second_check(self, image: Image.Image, siglip_top: List[str], part: str) -> Optional[Dict]:
        if not self.openai:
            return None
        ids = list(self.problems.keys())
        reference = "\n".join(f"- {p.id}: {p.name}. {p.key_signs}" for p in self.problems.values())
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "maize_second_check",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "is_maize": {"type": "string", "enum": ["yes", "no", "unclear"]},
                        "image_quality": {"type": "string", "enum": ["good", "limited", "unusable"]},
                        "plant_part": {"type": "string", "enum": ["leaf", "sheath", "stalk", "cob", "tassel", "whorl",
                                                                  "root", "whole_plant", "insect", "other"]},
                        "visible_signs": {"type": "array", "items": {"type": "string"}},
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "id": {"type": "string", "enum": ids},
                                    "likelihood": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "why": {"type": "string"},
                                },
                                "required": ["id", "likelihood", "why"],
                            },
                        },
                        "what_to_check": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["is_maize", "image_quality", "plant_part", "visible_signs", "candidates",
                                 "what_to_check"],
                },
            },
        }
        prompt = (
            "You are a maize plant-health expert following ICAR-IIMR descriptions. Look at the farmer's photo and "
            "list the 1-3 most likely problems from the list below, most likely first. Describe only what is "
            "visible. Use 'healthy' only if no symptom, pest or damage is visible anywhere in the photo. If symptoms "
            "are visible but no listed problem fits well, give your best matches with likelihood 'low'. "
            "If the photo shows an insect, larva, eggs or moth that is a maize pest, list it as a candidate even when "
            "it is photographed off the plant (on a hand, soil or another plant); set is_maize to 'unclear' then. "
            "what_to_check: 1-3 simple field checks a farmer can do to confirm (e.g. 'split the lower stalk and "
            "look for black specks').\n\nIIMR PROBLEMS:\n" + reference
        )
        buffer = io.BytesIO()
        small = image.copy()
        small.thumbnail((768, 768))
        small.save(buffer, "JPEG", quality=88)
        try:
            response = self.openai.chat.completions.create(
                model=self.second_check_model,
                temperature=0,
                response_format=schema,
                **({"reasoning_effort": os.getenv("SECOND_CHECK_REASONING", "none")}
                   if self.second_check_model.startswith(("gpt-5", "gpt-6")) else {}),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Image classifier's top guesses (may be wrong): "
                                                 f"{', '.join(siglip_top)}. Detected plant part: {part}."},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
                            "detail": "high"}},
                    ]},
                ],
            )
            result = json.loads(response.choices[0].message.content or "{}")
            usage = response.usage
            result["usage"] = {"input": usage.prompt_tokens, "output": usage.completion_tokens}
            return result
        except Exception as exc:  # noqa: BLE001 - degrade to SigLIP-only
            return {"error": f"{type(exc).__name__}: {exc}", "candidates": []}

    # --------------------------------------------------------------- diagnose
    def diagnose(self, image: Image.Image) -> Dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        siglip, parts = self.siglip(image)
        ranked = sorted(siglip.items(), key=lambda kv: -kv[1])
        top_id, top_p = ranked[0]
        part = max(parts, key=parts.get)

        needs_check = (
            top_p < CONFIDENT
            or top_id in CONFUSABLE
            or self.support.get(top_id, 0) < LOW_SUPPORT
            or part not in ("leaf", "insect")
        )
        check = self.second_check(image, [c for c, _ in ranked[:3]], part) if needs_check else None
        return self.fuse(siglip, parts, check)

    def fuse(self, siglip: Dict[str, float], parts: Dict[str, float], check: Optional[Dict]) -> Dict:
        ranked = sorted(siglip.items(), key=lambda kv: -kv[1])
        top_id, top_p = ranked[0]
        part = max(parts, key=parts.get)

        llm_scores: Dict[str, float] = {}
        for cand in (check or {}).get("candidates", [])[:3]:
            llm_scores[cand["id"]] = max(llm_scores.get(cand["id"], 0.0), LIKELIHOOD[cand["likelihood"]])
        if llm_scores:
            total = sum(llm_scores.values())
            llm_scores = {k: v / max(total, 1.0) for k, v in llm_scores.items()}
            leafy = part in ("leaf", "insect") and check.get("plant_part") in ("leaf", "whorl", "insect")
            w_siglip = 0.55 if leafy else 0.3
            scores = {}
            for cid in set(siglip) | set(llm_scores):
                s = siglip.get(cid, 0.0)
                if self.support.get(cid, 0) < LOW_SUPPORT:
                    s *= 0.5  # under-trained class: trust the head less
                scores[cid] = w_siglip * s + (1 - w_siglip) * llm_scores.get(cid, 0.0)
        else:
            scores = dict(siglip)
        fused = sorted(scores.items(), key=lambda kv: -kv[1])
        final_id, final_score = fused[0]
        llm_top = max(llm_scores, key=llm_scores.get) if llm_scores else None
        agree = llm_top is None or llm_top == final_id
        first = (check or {}).get("candidates") or [{}]
        llm_high = first[0].get("likelihood") == "high" and first[0].get("id") == final_id

        first_cand = ((check or {}).get("candidates") or [{}])[0]
        pest_photo = (check or {}).get("plant_part") == "insect" and first_cand.get("likelihood") in ("high", "medium") \
            and self.problems.get(first_cand.get("id"), Problem("", "", "", [], "", [])).category == "pest"
        if check and check.get("image_quality") == "unusable":
            status = "retake"
        elif check and check.get("is_maize") == "no" and not pest_photo and not (
                not (check.get("candidates")) and self.problems.get(top_id) is not None
                and self.problems[top_id].category == "pest" and top_p >= 0.6):
            # An insect photographed off the plant (e.g. a caterpillar in the hand) is still diagnosable.
            status = "retake"
        elif check is None and part == "not_maize" and parts["not_maize"] > 0.6 and top_p < 0.5:
            # Zero-shot "not maize" alone also fires on insect/gall close-ups; require an unsure head too.
            status = "retake"
        elif final_id == "healthy":
            llm_says_healthy = first_cand.get("id") == "healthy" and first_cand.get("likelihood") in ("high", "medium")
            healthy_ok = siglip.get("healthy", 0.0) >= HEALTHY_MIN and (check is None or llm_says_healthy)
            if healthy_ok:
                status = "confident"
            else:  # never call a plant healthy unless both models agree
                status = "uncertain"
                final_id = "no_clear_problem"  # shown as "not confirmed", never as a 0% disease
        elif check is None:
            status = "confident" if top_p >= CONFIDENT and top_id not in CONFUSABLE else "likely"
        elif not llm_scores:
            status = "likely" if top_p >= 0.6 else "uncertain"  # second check gave no opinion
        elif agree and (llm_high or siglip.get(final_id, 0.0) >= 0.6) and final_score >= LIKELY:
            status = "confident"
        elif final_score >= LIKELY or agree:
            status = "likely"
        else:
            status = "uncertain"

        problem = self.problems.get(final_id)
        advice = self.advice.get(final_id, {})
        category = problem.category if problem else "unknown"
        title = problem.name if problem else final_id
        if final_id == "no_clear_problem":
            title = "No clear problem visible (not confirmed)"
        return {
            "status": status,
            "problem_id": final_id,
            "title": title,
            "category": category,
            "tier": "review" if status in ("uncertain", "retake") else TIERS.get(category, "moderate"),
            "conf": round(100 * final_score, 1),
            "top3": [(self.problems[c].name if c in self.problems else c, round(100 * s, 1)) for c, s in fused[:3]],
            "top3_ids": [c for c, _ in fused[:3]],
            "agent": advice.get("causal_organism") or "",
            "signs": problem.key_signs if problem else "",
            "visible_signs": (check or {}).get("visible_signs", []),
            "what_to_check": (check or {}).get("what_to_check", []),
            "plant_part": (check or {}).get("plant_part") or part,
            "advice": advice,
            "siglip": {"top": top_id, "p": round(top_p, 3), "part": part, "part_p": round(parts[part], 3)},
            "second_check": None if check is None else {
                "model": self.second_check_model,
                "top": llm_top,
                "candidates": check.get("candidates", []),
                "error": check.get("error"),
                "usage": check.get("usage"),
            },
        }


def advice_text(result: Dict) -> Dict[str, List[str]]:
    """What the farmer should do, by status. Chemicals only when confident."""
    status = result["status"]
    advice = result.get("advice") or {}
    out: Dict[str, List[str]] = {"steps": [], "chemical": [], "source": list(advice.get("sources", []))}
    if status == "retake":
        out["steps"] = ["This does not look like a clear maize photo. Take a close, well-lit photo of the affected "
                        "leaf, stalk (split open if rotting), cob or tassel, filling most of the frame."]
        return out
    if result["problem_id"] == "healthy" and status == "confident":
        out["steps"] = ["No problem is visible. Keep scouting weekly, especially lower leaves, whorls and stalks."]
        return out
    if status == "uncertain":
        out["steps"] = ["The photo does not show enough to be sure. Take another photo close to the symptom "
                        "(both leaf surfaces, or the split lower stalk)."] + list(result.get("what_to_check", []))
        out["steps"].append("If many plants are affected, contact your nearest KVK or ICAR-IIMR before spraying.")
        return out
    management = list(advice.get("cultural", []))[:4] + list(advice.get("biological", []))[:2]
    if status == "likely":
        checks = list(result.get("what_to_check", [])) or [f"Confirm by checking: {result['signs']}"]
        out["steps"] = checks + management[:3]
        out["steps"].append("Confirm with your nearest KVK or ICAR-IIMR before any spray.")
        return out
    out["steps"] = management
    for item in advice.get("chemical", [])[:3]:
        detail = ", ".join(x for x in (item.get("method"), item.get("timing")) if x)
        out["chemical"].append(f"{item['product']} @ {item['dose']}" + (f" ({detail})" if detail else ""))
    if out["chemical"]:
        out["chemical"].append("Wear protective clothing and follow the label instructions.")
    if not out["steps"] and not out["chemical"]:
        out["steps"] = ["Contact your nearest KVK or ICAR-IIMR for management advice for this problem."]
    return out


if __name__ == "__main__":
    import sys

    diagnoser = MaizeDiagnoser()
    for path in sys.argv[1:]:
        result = diagnoser.diagnose(Image.open(path))
        print(json.dumps({k: v for k, v in result.items() if k != "advice"}, indent=2, default=str))
        print(json.dumps(advice_text(result), indent=2))
