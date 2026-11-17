"""Evaluate the full image pipeline on the held-out split (deck targets).

Uses the head trained on the train split only (models/iimr_head_v2_eval) and its
held-out list, so no evaluated image was seen in training. Reports, per class
and overall:
  * top-1 and correct-cause-in-top-3 accuracy (pests / diseases / nutrients)
  * sick plants called "healthy" (target 0)
  * status mix (confident / likely / uncertain / retake) and accuracy when confident
  * second-check rate, cost and latency
and compares: old production head | new head only | new head + second check.

Usage:
    python evals/run_image_eval.py [--per-class 25] [--no-second-check]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_head import OLD_NAME_MAP  # noqa: E402
from vlm_pipeline import MaizeDiagnoser  # noqa: E402

EVAL_HEAD = ROOT / "models" / "iimr_head_v2_eval"
RESULTS = ROOT / "evals" / "results"
PRICES = {"gpt-6-luna": (0.10, 0.50), "gpt-4.1-mini": (0.40, 1.60), "gpt-5.4-nano": (0.20, 1.25), "gpt-4o-mini": (0.15, 0.60),
          "gpt-5-mini": (0.25, 2.00), "gpt-4.1": (2.0, 8.0),
          "gpt-5.6-luna": (0.20, 1.20)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--no-second-check", action="store_true")
    parser.add_argument("--model", default=None, help="second-check model")
    parser.add_argument("--be-env", type=Path, default=None, help=".env holding OPENAI_API_KEY")
    args = parser.parse_args()
    if args.be_env:
        load_dotenv(args.be_env)

    rng = random.Random(7)
    split = json.loads((EVAL_HEAD / "test_split.json").read_text())
    by_label = defaultdict(list)
    for row in split:
        by_label[row["label"]].append(row)
    sample = []
    for label, rows in sorted(by_label.items()):
        rng.shuffle(rows)
        sample.extend(rows[: args.per_class])

    diagnoser = MaizeDiagnoser(head_dir=EVAL_HEAD, second_check_model=args.model,
                               enable_second_check=not args.no_second_check)
    old_clf = joblib.load(ROOT / "maize_vlm_classifier.joblib")
    old_scaler = joblib.load(ROOT / "maize_vlm_scaler.joblib")
    old_names = [OLD_NAME_MAP.get(n, n) for n in joblib.load(ROOT / "maize_class_names.joblib")]

    def run(row):
        image = Image.open(ROOT / row["path"]).convert("RGB")
        started = time.time()
        siglip, parts = diagnoser.siglip(image)
        ranked = sorted(siglip.items(), key=lambda kv: -kv[1])
        head_only = diagnoser.fuse(siglip, parts, None)
        result = diagnoser.diagnose(image)
        latency = time.time() - started
        # old production head (same SigLIP features, original image only)
        inputs = diagnoser.processor(images=[image], return_tensors="pt")
        import torch
        with torch.no_grad():
            tokens = diagnoser.model.vision_model(inputs["pixel_values"]).last_hidden_state
            feat = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
            feat = (feat / feat.norm(dim=-1, keepdim=True)).numpy()
        old_p = old_clf.predict_proba(old_scaler.transform(feat))[0]
        old_top = old_names[int(old_p.argmax())] if old_p.max() >= 0.6 else "healthy"
        old_top3 = [old_names[i] for i in np.argsort(-old_p)[:3]]
        return {
            "path": row["path"], "label": row["label"], "source": row["source"],
            "category": diagnoser.problems[row["label"]].category if row["label"] in diagnoser.problems else "?",
            "old_top1": old_top, "old_top3": old_top3,
            "head_top1": ranked[0][0], "head_top3": [c for c, _ in ranked[:3]], "head_status": head_only["status"],
            "final_top1": result["problem_id"], "final_top3": result["top3_ids"], "status": result["status"],
            "second_check": result["second_check"], "part": result["siglip"]["part"], "latency_s": round(latency, 2),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run, sample))

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = "head_only" if args.no_second_check else f"second_check_{diagnoser.second_check_model}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw").mkdir(exist_ok=True)
    (RESULTS / "raw" / f"{stamp}_{tag}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows))
    report = render(rows, tag, diagnoser.second_check_model)
    (RESULTS / f"{stamp}_{tag}.md").write_text(report)
    print(report)


def rate(rows, fn):
    return sum(1 for r in rows if fn(r)) / max(len(rows), 1)


def render(rows, tag, model) -> str:
    groups = {
        "Insect pests": [r for r in rows if r["category"] == "pest"],
        "Diseases": [r for r in rows if r["category"] == "disease"],
        "Nutrient deficiencies": [r for r in rows if r["category"] == "nutrient"],
        "Healthy": [r for r in rows if r["category"] == "healthy"],
        "All": rows,
    }
    lines = [f"# Maize VLM image eval: {tag}", "",
             f"{len(rows)} held-out images (never used to train the evaluated head), {datetime.now():%d %b %Y %H:%M}.",
             "", "| Group | n | Old head top-1 | Old top-3 | New head top-1 | New top-3 | Pipeline top-1 | Pipeline top-3 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in groups.items():
        if not group:
            continue
        lines.append(
            f"| {name} | {len(group)} | {rate(group, lambda r: r['old_top1'] == r['label']):.0%} "
            f"| {rate(group, lambda r: r['label'] in r['old_top3']):.0%} "
            f"| {rate(group, lambda r: r['head_top1'] == r['label']):.0%} "
            f"| {rate(group, lambda r: r['label'] in r['head_top3']):.0%} "
            f"| {rate(group, lambda r: r['final_top1'] == r['label']):.0%} "
            f"| {rate(group, lambda r: r['label'] in r['final_top3']):.0%} |")
    sick = [r for r in rows if r["label"] != "healthy"]
    lines += ["", "| Safety | Old head | Pipeline |", "|---|---:|---:|",
              f"| Sick plant called healthy | {sum(r['old_top1'] == 'healthy' for r in sick)} / {len(sick)} "
              f"| {sum(r['final_top1'] == 'healthy' for r in sick)} / {len(sick)} |"]
    confident = [r for r in rows if r["status"] == "confident"]
    lines += ["", "| Status | Share | Top-1 accuracy within status |", "|---|---:|---:|"]
    for status in ("confident", "likely", "uncertain", "retake"):
        group = [r for r in rows if r["status"] == status]
        if group:
            lines.append(f"| {status} | {len(group) / len(rows):.0%} | {rate(group, lambda r: r['final_top1'] == r['label']):.0%} |")
    checks = [r["second_check"] for r in rows if r["second_check"]]
    if checks:
        price_in, price_out = PRICES.get(model, (0.4, 1.6))
        costs = [((c.get("usage") or {}).get("input", 0) * price_in + (c.get("usage") or {}).get("output", 0) * price_out) / 1e6
                 for c in checks]
        lines += ["", f"Second check ran on {len(checks)}/{len(rows)} images ({len(checks) / len(rows):.0%}); "
                      f"mean cost per checked image ${statistics.mean(costs):.4f}; "
                      f"errors {sum(1 for c in checks if c.get('error'))}."]
    lat = sorted(r["latency_s"] for r in rows)
    lines.append(f"Latency per image (CPU, incl. second check): p50 {statistics.median(lat):.1f}s, "
                 f"p95 {lat[int(0.95 * (len(lat) - 1))]:.1f}s.")
    lines += ["", "## Per class (pipeline)", "", "| Class | n | top-1 | top-3 | confident share |", "|---|---:|---:|---:|---:|"]
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
    for label, group in sorted(by_class.items()):
        lines.append(f"| {label} | {len(group)} | {rate(group, lambda r: r['final_top1'] == r['label']):.0%} "
                     f"| {rate(group, lambda r: r['label'] in r['final_top3']):.0%} "
                     f"| {rate(group, lambda r: r['status'] == 'confident'):.0%} |")
    if confident:
        lines.append(f"\nAccuracy when the app says 'confident': {rate(confident, lambda r: r['final_top1'] == r['label']):.0%} "
                     f"({len(confident)} images).")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
