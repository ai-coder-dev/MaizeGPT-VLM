"""Second-pass quality filter for photos harvested from ICAR-IIMR PDFs.

A visual spot-check showed the caption-based labels include diagrams, disease
cycles, micrographs, petri dishes, scoring-scale charts, placeholder graphics
and wide field/people shots. This pass keeps a photo only if it is a real photo
of a maize plant part / insect AND the labelled problem's signs are visible.
All IIMR-PDF "healthy" photos are dropped (mostly machinery, grain, people).

Usage:
    python training/filter_iimr_photos.py --be-env "<MaizeGPT-BE>/.env"
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PROBLEMS = {p["id"]: p for p in json.loads((ROOT / "knowledge" / "iimr_problems.json").read_text())["problems"]}
MODEL = os.getenv("FILTER_MODEL", "gpt-6-luna")

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "photo_filter",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["closeup_plant_or_insect_photo", "field_overview_photo", "diagram_or_chart",
                                                    "micrograph_or_lab", "people_or_machinery", "grain_or_product",
                                                    "placeholder_or_blank", "other"]},
                "signs_visible": {"type": "string", "enum": ["yes", "no", "unsure"]},
            },
            "required": ["kind", "signs_visible"],
        },
    },
}


def judge(client: OpenAI, row: dict) -> dict:
    problem = PROBLEMS.get(row["label"], {"name": row["label"], "key_signs": ""})
    image = Image.open(ROOT / row["path"]).convert("RGB")
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format=SCHEMA,
        **({"reasoning_effort": "none"} if MODEL.startswith(("gpt-5", "gpt-6")) else {}),
        messages=[
            {"role": "system", "content": (
                "You screen images for a maize disease/pest training set. Classify what kind of image it is. "
                "signs_visible: 'yes' only if the image clearly shows the signs of the named problem on maize."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": f"Named problem: {problem['name']}. Typical signs: {problem['key_signs']}"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(), "detail": "low"}},
            ]},
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--be-env", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(args.be_env)
    client = OpenAI()

    with MANIFEST.open() as handle:
        rows = list(csv.DictReader(handle))
    candidates = [r for r in rows if r["source"] == "iimr_pdf" and r["label"] != "healthy"]

    def work(row):
        try:
            return row, judge(client, row)
        except Exception as exc:  # noqa: BLE001
            return row, {"error": str(exc)}

    with ThreadPoolExecutor(max_workers=12) as pool:
        verdicts = dict((r["path"], v) for r, v in pool.map(work, candidates))

    errors = [v for v in verdicts.values() if "error" in v]
    if errors:
        # Never delete data because of an API failure (e.g. quota exhausted).
        raise SystemExit(f"{len(errors)} API errors (first: {errors[0]['error'][:200]}); nothing changed.")

    # Rejected photos are never deleted: they move to data/rejected/<label>/ and are
    # listed with the reason in data/rejected.csv, so they can be reviewed or restored.
    rejected_csv = ROOT / "data" / "rejected.csv"
    rejected_rows = []
    if rejected_csv.exists():
        with rejected_csv.open() as handle:
            rejected_rows = list(csv.DictReader(handle))
    keep, moved = [], 0
    for row in rows:
        if row["source"] != "iimr_pdf":
            keep.append(row)
            continue
        verdict = verdicts.get(row["path"], {})
        ok = (row["label"] != "healthy" and verdict.get("kind") == "closeup_plant_or_insect_photo"
              and verdict.get("signs_visible") == "yes")
        if ok:
            keep.append(row)
            continue
        source = ROOT / row["path"]
        target = ROOT / "data" / "rejected" / row["label"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            source.replace(target)
        reason = ("healthy photos from PDFs are not used" if row["label"] == "healthy"
                  else f"kind={verdict.get('kind')}, signs_visible={verdict.get('signs_visible')}")
        rejected_rows.append({**row, "path": str(target.relative_to(ROOT)), "reason": reason})
        moved += 1
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(keep)
    if rejected_rows:
        with rejected_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rejected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rejected_rows)
    kept = [r for r in keep if r["source"] == "iimr_pdf"]
    print(f"kept {len(kept)} IIMR photos for training; moved {moved} to data/rejected/ (see data/rejected.csv)")
    from collections import Counter
    for label, n in Counter(r["label"] for r in kept).most_common():
        print(f"  {n:4} {label}")


if __name__ == "__main__":
    main()
