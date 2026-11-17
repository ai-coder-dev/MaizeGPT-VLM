"""Export ICAR-IIMR management advice for each VLM problem from the MaizeGPT table.

MaizeGPT-BE's knowledge/tables/pests_diseases.json is extracted from ICAR-IIMR
publications and grounding-checked (and agronomist-verified where marked). This
script maps its rows onto the VLM's problem ids and writes knowledge/advice.json,
so the image app and the chatbot give the same, sourced advice.

Usage:
    python training/export_advice.py --table "<MaizeGPT-BE>/knowledge/tables/pests_diseases.json" [--verified-only]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge" / "advice.json"
CATEGORY = {p["id"]: p["category"] for p in json.loads((ROOT / "knowledge" / "iimr_problems.json").read_text())["problems"]}

KEYWORDS = {
    "maydis_leaf_blight": ["maydis leaf blight", "southern leaf blight", "bipolaris maydis", "mlb"],
    "turcicum_leaf_blight": ["turcicum leaf blight", "northern leaf blight", "northern corn leaf blight",
                             "exserohilum turcicum", "tlb"],
    "banded_leaf_sheath_blight": ["banded leaf and sheath blight", "banded leaf & sheath blight", "rhizoctonia", "blsb"],
    "charcoal_rot": ["charcoal rot", "macrophomina"],
    "fusarium_stalk_rot": ["fusarium stalk rot", "fusarium verticillioides stalk"],
    "bacterial_stalk_rot": ["bacterial stalk rot", "dickeya", "erwinia"],
    "pythium_stalk_rot": ["pythium stalk rot", "pythium"],
    "common_rust": ["common rust", "puccinia sorghi"],
    "polysora_rust": ["polysora rust", "southern rust", "puccinia polysora"],
    "grey_leaf_spot": ["grey leaf spot", "gray leaf spot", "cercospora"],
    "downy_mildew": ["downy mildew", "peronosclerospora", "sclerophthora"],
    "common_smut": ["common smut", "ustilago maydis"],
    "head_smut": ["head smut", "sphacelotheca", "sporisorium"],
    "ear_rot": ["ear rot", "cob rot", "aspergillus", "aflatoxin"],
    "curvularia_leaf_spot": ["curvularia"],
    "phaeosphaeria_leaf_spot": ["phaeosphaeria"],
    "bacterial_leaf_streak": ["bacterial leaf streak", "xanthomonas"],
    "viral_disease": ["mosaic", "virus", "viral"],
    "fall_armyworm": ["fall armyworm", "spodoptera frugiperda"],
    "stem_borer": ["spotted stem borer", "chilo partellus", "maize stem borer", "stem borer"],
    "pink_stem_borer": ["pink stem borer", "sesamia inferens", "sesamia"],
    "shoot_fly": ["shoot fly", "atherigona"],
    "maize_aphid": ["aphid", "rhopalosiphum"],
    "thrips": ["thrips"],
    "termite": ["termite"],
    "corn_earworm": ["helicoverpa", "cob borer", "earworm", "ear head bug"],
    "armyworm_cutworm": ["cutworm", "agrotis", "mythimna", "oriental armyworm"],
    "nitrogen_deficiency": ["nitrogen deficiency"],
    "phosphorus_deficiency": ["phosphorus deficiency"],
    "potassium_deficiency": ["potassium deficiency"],
    "zinc_deficiency": ["zinc deficiency", "white bud"],
}

# Active-ingredient lexicon by use. A chemical entry is kept only if its product names
# an active of the right type for the problem and its dose contains a number.
INSECTICIDES = ["chlorantraniliprole", "chlorantriniliprole", "emamectin", "spinetoram", "spinosad", "thiamethoxam",
                "imidacloprid", "clothianidin", "fipronil", "lambda", "cyhalothrin", "cypermethrin", "deltamethrin",
                "chlorpyrifos", "chloropyrifos", "quinalphos", "dimethoate", "profenofos", "flubendiamide", "flubendamide",
                "novaluron", "novaluran", "indoxacarb", "tetraniliprole", "cyantraniliprole", "broflanilide", "acephate",
                "malathion", "thiodicarb", "bifenthrin", "acetamiprid", "azadirachtin", "neem", "nske", "bacillus thuringiensis",
                "metarhizium", "beauveria", "nomuraea", "trichogramma", "fluopyram", "carbofuran", "monocrotophos"]
FUNGICIDES = ["mancozeb", "carbendazim", "propiconazole", "tebuconazole", "tabuconazole", "azoxystrobin", "difenoconazole",
              "difenconazole", "hexaconazole", "validamycin", "metalaxyl", "mefenoxam", "thiram", "captan", "trifloxystrobin",
              "pyraclostrobin", "copper", "streptocycline", "streptomycin", "bleaching powder", "zineb", "dithane", "ziram",
              "carboxin", "fludioxonil", "kresoxim", "trichoderma", "pseudomonas", "salicylic", "bavistin", "thiophanate",
              "apron", "ridomil", "amistar", "tilt", "folicur"]
NUTRIENTS = ["zinc", "znso4", "urea", "dap", "nitrogen", "phosph", "potash", "potassium", "mop", "ssp", "12:61", "13:0:45",
             "npk", "sulphate", "sulfate"]


# Statements about breeding material or trials are not farmer actions.
RESEARCH_FINDING = re.compile(r"\b(inbreds?|inbred lines?|genotypes?|germplasm|CML ?\d|NBPGR|registered|were found|"
                              r"was found|screening|screened|accessions?|entries|donors?|QTL|markers?)\b", re.IGNORECASE)

ALIASES = {"chlorantriniliprole": "chlorantraniliprole", "difenconazole": "difenoconazole", "tabuconazole": "tebuconazole",
           "flubendamide": "flubendiamide", "novaluran": "novaluron", "chloropyrifos": "chlorpyrifos",
           "amistar": "azoxystrobin+difenoconazole", "indofil": "mancozeb", "dithane": "zineb-or-mancozeb",
           "bavistin": "carbendazim", "sheethmar": "validamycin"}


def active_key(product: str) -> str:
    """Canonical set of actives in a product, used to de-duplicate spellings and brand names."""
    text = product.lower()
    found = set()
    for word in INSECTICIDES + FUNGICIDES + NUTRIENTS + list(ALIASES):
        if word in text:
            found.add(ALIASES.get(word, word))
    expanded = set()
    for item in found:
        expanded.update(item.split("+"))
    return "+".join(sorted(expanded))


def chemical_ok(category: str, product: str, dose: str) -> bool:
    text = product.lower()
    if not re.search(r"\d", dose or ""):
        return False
    allowed = {"pest": INSECTICIDES, "disease": FUNGICIDES, "nutrient": NUTRIENTS}.get(category, INSECTICIDES + FUNGICIDES)
    return any(word in text for word in allowed)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9& ]", " ", str(text).lower())).strip()


def best_problem(row: dict):
    haystack = norm(" ".join([row.get("name", ""), row.get("causal_organism", "")] + row.get("aliases", [])))
    best, best_len = None, 0
    for pid, words in KEYWORDS.items():
        for word in words:
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, haystack) and len(word) > best_len:
                best, best_len = pid, len(word)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--verified-only", action="store_true")
    args = parser.parse_args()

    policy_path = args.table.parent / "chemical_policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    cleared = {k.lower() for k in policy.get("cleared", {})}
    blocked = {k.lower() for k in policy.get("banned", {})} | {
        k.lower() for k in policy.get("verify_before_use", {}) if k.lower() not in cleared}

    def is_blocked(text: str) -> bool:
        low = text.lower().replace("-", " ")
        return any(re.search(rf"\b{re.escape(b)}\b", low) for b in blocked)

    def year_of(row: dict) -> int:
        match = re.search(r"(19|20)\d{2}", str(row.get("source_year", "")))
        return int(match.group(0)) if match else 0

    rows = json.loads(args.table.read_text())
    # Newest publications first, so the latest IIMR recommendation wins on conflicts.
    rows.sort(key=year_of, reverse=True)
    advice: dict = {}
    for row in rows:
        if args.verified_only and not row.get("verified"):
            continue
        if not args.verified_only and not (row.get("verified") or row.get("auto_check") == "passed"):
            continue
        pid = best_problem(row)
        if not pid:
            continue
        entry = advice.setdefault(pid, {"causal_organism": "", "cultural": [], "biological": [], "chemical": [],
                                        "etl": "", "sources": [], "verified": True})
        entry["causal_organism"] = entry["causal_organism"] or row.get("causal_organism", "")
        entry["etl"] = entry["etl"] or row.get("etl", "")
        entry["verified"] = entry["verified"] and bool(row.get("verified"))
        for key, field in (("cultural", "cultural_management"), ("biological", "biological_management")):
            for item in row.get(field, []):
                if norm(item) and norm(item) not in {norm(x) for x in entry[key]} and not is_blocked(item) \
                        and not RESEARCH_FINDING.search(item):
                    entry[key].append(item)
        for chem in row.get("chemical_management", []):
            text = f"{chem.get('product', '')} {chem.get('dose', '')}"
            if is_blocked(text):
                entry.setdefault("blocked", []).append(chem.get("product", ""))
                continue
            if not chemical_ok(CATEGORY.get(pid, ""), chem.get("product", ""), chem.get("dose", "")):
                entry.setdefault("rejected_entries", []).append(text.strip())
                continue
            actives = active_key(chem.get("product", ""))
            if actives and actives not in {active_key(c["product"]) for c in entry["chemical"]}:
                entry["chemical"].append({**{k: chem.get(k, "") for k in ("product", "dose", "method", "timing")},
                                          "source": f"{row.get('source_title')} ({row.get('source_year')}), p. {row.get('page')}"})
        source = f"ICAR-IIMR, {row.get('source_title')} ({row.get('source_year')}), p. {row.get('page')}"
        if source not in entry["sources"]:
            entry["sources"].append(source)

    devanagari = re.compile(r"[\u0900-\u097F]")
    for entry in advice.values():
        # English UI: English entries first (Hindi kept after them), newest sources already first.
        # Merge "IMH 225 is moderately resistant to X." statements into one actionable line.
        resistant, rest = [], []
        for item in entry["cultural"]:
            match = re.match(r"^\s*([A-Z][A-Za-z]*[\s-]?[A-Z]*\s?-?\d[\w-]*)\s+is\s+(?:moderately\s+)?resistant", item)
            (resistant if match else rest).append(match.group(1).strip() if match else item)
        if resistant:
            rest.insert(0, "Grow hybrids reported as resistant or moderately resistant: "
                        + ", ".join(dict.fromkeys(resistant)) + ".")
        entry["cultural"] = rest
        for key in ("cultural", "biological"):
            entry[key] = sorted(entry[key], key=lambda t: bool(devanagari.search(t)))[:8]
        entry["chemical"] = sorted(entry["chemical"], key=lambda c: bool(devanagari.search(c["product"])))[:5]
        entry["sources"] = entry["sources"][:4]
        entry["blocked"] = sorted(set(entry.get("blocked", [])))
        entry["rejected_entries"] = sorted(set(entry.get("rejected_entries", [])))
    OUT.write_text(json.dumps(advice, indent=2, ensure_ascii=False))
    print(f"advice for {len(advice)} problems -> {OUT.relative_to(ROOT)}")
    for pid in KEYWORDS:
        e = advice.get(pid)
        print(f"  {pid:28} {'-' if not e else f'{len(e['chemical'])} chem, {len(e['cultural'])} cultural, {len(e['sources'])} src'}")


if __name__ == "__main__":
    main()
