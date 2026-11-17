"""Assemble a multi-source maize image dataset mapped to the IIMR problem list.

Every image is resized to <=512px, saved under data/images/<class>/ and logged in
data/manifest.csv with its source, licence and attribution URL. Images are pulled
member-by-member from remote zips (HTTP range requests), so the multi-GB archives
are never downloaded in full.

Sources (all licences allow commercial use with attribution):
  * agronomist  - RootsGoods agronomist-labelled photos (MaizeGPT-BE/test_data).
                  These were the original training images; TRAIN ONLY, never eval.
  * mendeley    - Corn Leaf Disease Classification Dataset, 10.17632/hmkd6nbngr.1, CC BY 4.0
  * pretoria    - "Diseases of maize in the field", Univ. of Pretoria, figshare 20237613, CC BY 4.0
  * inat        - iNaturalist research-grade observations with CC0 / CC BY / CC BY-SA photos

Ambiguous upstream labels are skipped (Mendeley "Corn Leaf Blight" mixes MLB and
TLB; "Insects damage" has no pest identity; Pretoria "Other"/"Unknown").

Usage:
    python training/build_dataset.py --agronomist-dir "<path>/MaizeGPT-BE/test_data" \
        [--per-class 400] [--sources agronomist,mendeley,pretoria,inat]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imagehash
import requests
from PIL import Image, ImageOps
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
MANIFEST = DATA / "manifest.csv"
MAX_SIDE = 512
SEED = 20261117
HEADERS = {"User-Agent": "RootsGoods-MaizeVLM-dataset/1.0"}

MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/hmkd6nbngr/files/"
    "fe0afde4-b436-45e5-a106-975f91d7da5c/file_downloaded"
)
MENDELEY_MAP = {"Common Rust": "common_rust", "Gray Leaf Spot": "grey_leaf_spot", "Healthy": "healthy"}

PRETORIA_ZIP = "https://ndownloader.figshare.com/files/36203652"
PRETORIA_CSV = "https://ndownloader.figshare.com/files/36203595"
PRETORIA_MAP = {
    1: "grey_leaf_spot",
    2: "turcicum_leaf_blight",
    3: "phaeosphaeria_leaf_spot",
    4: "common_rust",
    5: "polysora_rust",
    6: "healthy",
}

AGRONOMIST_MAP = {
    "fall army worm": "fall_armyworm",
    "stem borer": "stem_borer",
    "trips": "thrips",
    "northern leaf blight": "turcicum_leaf_blight",
    "maydis leaf blight": "maydis_leaf_blight",
    "grey leaf spot": "grey_leaf_spot",
    "rust in maize": "common_rust",
    "charcoal rot": "charcoal_rot",
    "nitrogen deficiency in maize": "nitrogen_deficiency",
    "phosphorus deficiency": "phosphorus_deficiency",
    "potassium deficiency in maize": "potassium_deficiency",
    "zinc deficiency in maize": "zinc_deficiency",
}

# iNaturalist taxa. `maize_only` restricts to observations annotated/described
# on Zea mays where the taxon has many hosts.
INAT_TAXA = {
    "common_smut": {"taxon_name": "Ustilago maydis"},
    "polysora_rust": {"taxon_name": "Puccinia polysora"},
    "common_rust": {"taxon_name": "Puccinia sorghi"},
    "turcicum_leaf_blight": {"taxon_name": "Exserohilum turcicum"},
    "maydis_leaf_blight": {"taxon_name": "Bipolaris maydis"},
    "grey_leaf_spot": {"taxon_name": "Cercospora zeae-maydis"},
    "fall_armyworm": {"taxon_name": "Spodoptera frugiperda"},
    "maize_aphid": {"taxon_name": "Rhopalosiphum maidis"},
    "head_smut": {"taxon_name": "Sporisorium reilianum"},
    "downy_mildew": {"taxon_name": "Peronosclerospora"},
    "stem_borer": {"taxon_name": "Chilo partellus"},
    "pink_stem_borer": {"taxon_name": "Sesamia inferens"},
    "corn_earworm": {"taxon_name": "Helicoverpa armigera"},
}
INAT_LICENSES = "cc0,cc-by,cc-by-sa"


def resize(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


class Writer:
    """Saves images, de-duplicates by perceptual hash, and records provenance."""

    def __init__(self) -> None:
        IMAGES.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, str]] = []
        self.hashes: Dict[str, str] = {}
        self.counts: Dict[Tuple[str, str], int] = defaultdict(int)
        if MANIFEST.exists():
            with MANIFEST.open() as handle:
                for row in csv.DictReader(handle):
                    self.rows.append(row)
                    self.hashes[row["phash"]] = row["path"]
                    self.counts[(row["source"], row["label"])] += 1

    def have(self, source: str, label: str) -> int:
        return self.counts[(source, label)]

    def add(self, image: Image.Image, label: str, source: str, source_id: str,
            license_name: str, attribution: str, group: str) -> bool:
        image = resize(image)
        phash = str(imagehash.phash(image))
        if phash in self.hashes:
            return False
        name = f"{source}__{hashlib.sha1(source_id.encode()).hexdigest()[:12]}.jpg"
        path = IMAGES / label / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, "JPEG", quality=90)
        self.hashes[phash] = str(path)
        self.counts[(source, label)] += 1
        self.rows.append({
            "path": str(path.relative_to(ROOT)),
            "label": label,
            "source": source,
            "source_id": source_id,
            "group": group,
            "license": license_name,
            "attribution": attribution,
            "phash": phash,
        })
        return True

    def save(self) -> None:
        with MANIFEST.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


def from_agronomist(writer: Writer, folder: Path) -> None:
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        raw = re.sub(r"^\d+\.\s*", "", class_dir.name).strip().lower()
        label = AGRONOMIST_MAP.get(raw)
        if not label:
            print(f"  skip agronomist folder {class_dir.name}")
            continue
        for file in sorted(class_dir.iterdir()):
            try:
                image = Image.open(file)
            except Exception:  # noqa: BLE001
                continue
            writer.add(image, label, "agronomist", f"{class_dir.name}/{file.name}",
                       "RootsGoods internal", "RootsGoods agronomist", f"agronomist/{file.name}")


def resolve(url: str) -> str:
    return requests.head(url, allow_redirects=True, headers=HEADERS, timeout=60).url


def from_mendeley(writer: Writer, per_class: int) -> None:
    rng = random.Random(SEED)
    with RemoteZip(resolve(MENDELEY_URL)) as archive:
        by_label = defaultdict(list)
        for name in archive.namelist():
            folder = name.split("/")[-2] if "/" in name else ""
            if folder in MENDELEY_MAP and not name.endswith("/"):
                by_label[MENDELEY_MAP[folder]].append(name)
        for label, names in by_label.items():
            rng.shuffle(names)
            for name in names:
                if writer.have("mendeley", label) >= per_class:
                    break
                try:
                    image = Image.open(io.BytesIO(archive.read(name)))
                except Exception:  # noqa: BLE001
                    continue
                writer.add(image, label, "mendeley", name, "CC BY 4.0",
                           "Rahman, Corn Leaf Disease Classification Dataset, Mendeley Data, 10.17632/hmkd6nbngr.1",
                           f"mendeley/{name}")
            print(f"  mendeley {label}: {writer.have('mendeley', label)}")


def from_pretoria(writer: Writer, per_class: int) -> None:
    rng = random.Random(SEED)
    # Database.csv is one-hot: imgID_id,filePath,GLS,NCLB,PLS,CR,SR,NoFoliarSymptoms,Other,UnidentifiedDisease
    columns = {"GLS": 1, "NCLB": 2, "PLS": 3, "CR": 4, "SR": 5, "NoFoliarSymptoms": 6}
    text = requests.get(PRETORIA_CSV, headers=HEADERS, timeout=60).text
    single = defaultdict(list)
    for row in csv.DictReader(io.StringIO(text)):
        positives = [col for col in ("GLS", "NCLB", "PLS", "CR", "SR", "NoFoliarSymptoms", "Other",
                                     "UnidentifiedDisease") if str(row.get(col, "0")).strip() == "1"]
        if len(positives) == 1 and positives[0] in columns:
            single[PRETORIA_MAP[columns[positives[0]]]].append(row["filePath"].strip())
    # figshare's signed S3 URL expires after a few minutes: reopen on failure.
    archive = RemoteZip(resolve(PRETORIA_ZIP))
    lookup = {name.rsplit("/", 1)[-1]: name for name in archive.namelist() if not name.endswith("/")}
    for label, paths in single.items():
        rng.shuffle(paths)
        failures = 0
        for path in paths:
            if writer.have("pretoria", label) >= per_class or failures > 20:
                break
            member = lookup.get(path.rsplit("/", 1)[-1])
            if not member:
                continue
            for attempt in range(2):
                try:
                    image = Image.open(io.BytesIO(archive.read(member)))
                    image.load()
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 0:
                        archive.close()
                        archive = RemoteZip(resolve(PRETORIA_ZIP))
                    else:
                        failures += 1
                        print(f"    ! {member}: {exc}")
                        image = None
            if image is None:
                continue
            # File names look like 110_CIMG3846_1-109-1.JPG: "1-109" identifies the plant,
            # the last digit the view. Group by plant so CV never splits one plant.
            parts = path.rsplit(".", 1)[0].split("_")
            plant = parts[-1].rsplit("-", 1)[0] if len(parts) >= 3 else path
            writer.add(image, label, "pretoria", path, "CC BY 4.0",
                       "University of Pretoria, Diseases of maize in the field, figshare 20237613",
                       f"pretoria/{plant}")
        print(f"  pretoria {label}: {writer.have('pretoria', label)}")
    archive.close()


def inat_photos(params: Dict[str, str], limit: int) -> Iterable[Dict]:
    page = 1
    seen = 0
    while seen < limit:
        query = {
            **params,
            "quality_grade": "research",
            "photo_license": INAT_LICENSES,
            "per_page": 100,
            "page": page,
            "order_by": "votes",
        }
        response = requests.get("https://api.inaturalist.org/v1/observations", params=query,
                                headers=HEADERS, timeout=60)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return
        for observation in results:
            for photo in observation.get("photos", [])[:2]:
                if (photo.get("license_code") or "").lower() not in INAT_LICENSES.split(","):
                    continue
                seen += 1
                yield {
                    "observation": observation["id"],
                    "photo": photo["id"],
                    "url": photo["url"].replace("square", "medium"),
                    "license": photo["license_code"],
                    "attribution": photo.get("attribution", ""),
                    "user": (observation.get("user") or {}).get("login", ""),
                }
        page += 1
        time.sleep(1.0)  # iNaturalist asks for <= 1 request/second


def from_inat(writer: Writer, per_class: int) -> None:
    for label, params in INAT_TAXA.items():
        try:
            for item in inat_photos(params, per_class * 2):
                if writer.have("inat", label) >= per_class:
                    break
                try:
                    content = requests.get(item["url"], headers=HEADERS, timeout=60).content
                    image = Image.open(io.BytesIO(content))
                except Exception:  # noqa: BLE001
                    continue
                writer.add(image, label, "inat", f"{item['observation']}/{item['photo']}",
                           item["license"], f"{item['attribution']} (iNaturalist obs {item['observation']})",
                           f"inat/{item['user']}")
        except requests.RequestException as exc:
            print(f"  ! inat {label}: {exc}")
        print(f"  inat {label}: {writer.have('inat', label)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agronomist-dir", type=Path)
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--sources", default="agronomist,mendeley,pretoria,inat")
    args = parser.parse_args()

    writer = Writer()
    sources = args.sources.split(",")
    try:
        if "agronomist" in sources and args.agronomist_dir:
            print("agronomist")
            from_agronomist(writer, args.agronomist_dir)
        if "mendeley" in sources:
            print("mendeley")
            from_mendeley(writer, args.per_class)
        if "pretoria" in sources:
            print("pretoria")
            from_pretoria(writer, args.per_class)
        if "inat" in sources:
            print("inat")
            from_inat(writer, min(args.per_class, 250))
    finally:
        if writer.rows:
            writer.save()

    totals = defaultdict(int)
    for row in writer.rows:
        totals[row["label"]] += 1
    print("\nimages per class:")
    for label, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5}  {label}")


if __name__ == "__main__":
    main()
