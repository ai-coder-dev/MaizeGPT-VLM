"""Harvest expert-captioned symptom photos from ICAR-IIMR PDFs.

ICAR-IIMR bulletins contain field photos of diseases and pests that the open
datasets lack (BLSB, stalk rots, downy mildew, smuts, borers...). This script
extracts embedded photos from the bulletins downloaded by MaizeGPT-BE
(knowledge/raw/) and labels each one with the IIMR problem id ONLY when the
page caption clearly names it. Uncertain images are discarded.

Labelled images are added to data/images/ with source "iimr_pdf" and grouped by
publication, so cross-validation never splits one bulletin across folds.

Usage:
    python training/extract_pdf_figures.py --raw-manifest "<MaizeGPT-BE>/knowledge/raw/manifest.json"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from build_dataset import Writer  # noqa: E402

PROBLEMS = json.loads((ROOT / "knowledge" / "iimr_problems.json").read_text())["problems"]
PROBLEM_IDS = [p["id"] for p in PROBLEMS]
SOURCE_PATTERN = re.compile(r"disease|pest|armyworm|aflatoxin|screening|integrated|production|rabi|storage|"
                            r"रोग|कीट|सड़न|सडन", re.IGNORECASE)
MIN_SIDE = 220
MODEL = os.getenv("FIGURE_LABEL_MODEL", "gpt-4.1-mini")

LABEL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "figure_label",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_maize_photo": {"type": "boolean"},
                "label": {"type": "string", "enum": [*PROBLEM_IDS, "uncertain"]},
                "caption_quote": {"type": "string"},
            },
            "required": ["is_maize_photo", "label", "caption_quote"],
        },
    },
}

PROMPT = """You label photos extracted from an ICAR-IIMR maize publication for a training dataset.
- is_maize_photo: true only for a real photograph of maize plants/parts/insects in the field or lab (not a diagram, chart, logo, map, or photo of people/machinery).
- label: the problem shown, chosen ONLY when the page text captions this photo (e.g. "Figure 3. Symptoms of BLSB") or the page is clearly about exactly one problem AND the photo visibly matches it. If the page shows several problems and you cannot tell which caption belongs to this photo, return "uncertain".
- caption_quote: the caption/heading text you relied on (verbatim), or ''.
Problem ids: """ + ", ".join(f"{p['id']} = {p['name']}" for p in PROBLEMS)


def page_photos(page: pymupdf.Page) -> List[Image.Image]:
    photos = []
    page_area = page.rect.width * page.rect.height
    for info in page.get_images(full=True):
        xref = info[0]
        rects = page.get_image_rects(xref)
        if rects and (rects[0].width * rects[0].height) > 0.85 * page_area:
            continue  # full-page scan: cannot isolate individual photos
        try:
            pix = pymupdf.Pixmap(page.parent, xref)
            if pix.n - pix.alpha >= 4:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        if min(image.size) >= MIN_SIDE:
            photos.append(image)
    return photos


def label(client: OpenAI, image: Image.Image, page_text: str) -> Optional[Dict]:
    thumb = image.copy()
    thumb.thumbnail((512, 512))
    buffer = io.BytesIO()
    thumb.save(buffer, "JPEG", quality=85)
    response = client.chat.completions.create(
        model=MODEL, temperature=0, response_format=LABEL_SCHEMA,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"PAGE TEXT:\n{page_text[:3500]}"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(), "detail": "low"}},
            ]},
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(args.raw_manifest.parents[2] / ".env")
    client = OpenAI()
    be_root = args.raw_manifest.parents[2]
    writer = Writer()

    jobs = []
    for item in json.loads(args.raw_manifest.read_text()):
        if item.get("status") != "ok" or not item["path"].endswith(".pdf"):
            continue
        if not SOURCE_PATTERN.search(item["title"]):
            continue
        doc = pymupdf.open(be_root / item["path"])
        for page in doc:
            text = page.get_text()
            for index, photo in enumerate(page_photos(page)):
                jobs.append((item, page.number + 1, index, photo, text))
    print(f"{len(jobs)} candidate photos")

    def work(job):
        item, page_number, index, photo, text = job
        try:
            return job, label(client, photo, text)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {item['title'][:40]} p{page_number}: {exc}")
            return job, None

    kept = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (item, page_number, index, photo, _), result in pool.map(work, jobs):
            if not result or not result.get("is_maize_photo") or result.get("label") in (None, "uncertain"):
                continue
            if writer.add(photo, result["label"], "iimr_pdf", f"{item['id']}/p{page_number}/{index}",
                          "ICAR-IIMR publication (shared by ICAR-IIMR for this project)",
                          f"{item['title']} ({item.get('year', '')}), p.{page_number}: {result.get('caption_quote', '')[:120]}",
                          f"iimr_pdf/{item['id']}"):
                kept += 1
    writer.save()
    print(f"kept {kept} labelled photos")


if __name__ == "__main__":
    main()
