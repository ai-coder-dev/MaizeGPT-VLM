"""Retrain the SigLIP classifier head on the IIMR problem list.

Steps
1. Extract SigLIP (base, patch16-384) features exactly as at inference (mean +
   max token pooling, L2-normalised) for every image in data/manifest.csv, plus a
   horizontally-flipped copy for augmentation. Cached in features/.
2. Hold out ~25% of image GROUPS per class (source session / contributor /
   publication) as an evaluation split. Agronomist photos (the old model's
   training data) are never used for evaluation.
3. Fit StandardScaler + class-balanced LogisticRegression, then fit a softmax
   temperature on the held-out split for calibrated confidence.
4. Report per-class accuracy for the new head AND the current production head on
   the same held-out images, then refit on all data and save the artifacts.

Usage:
    python training/train_head.py [--min-images 10] [--C 0.5]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
FEATURES = ROOT / "features" / "siglip_base_384.npz"
ARTIFACTS = ROOT / "models" / "iimr_head_v2"
EVAL_ARTIFACTS = ROOT / "models" / "iimr_head_v2_eval"
VISION_ID = "google/siglip-base-patch16-384"
OLD_NAME_MAP = {
    "Charcoal_Rot": "charcoal_rot", "Fall_Army_worm": "fall_armyworm", "Grey_Leaf_Spot": "grey_leaf_spot",
    "Maydis_Leaf_Blight": "maydis_leaf_blight", "Nitrogen_Deficiency_in_Maize": "nitrogen_deficiency",
    "Northern_Leaf_Blight": "turcicum_leaf_blight", "Phosphorus_Deficiency": "phosphorus_deficiency",
    "Potassium_Deficiency_in_Maize": "potassium_deficiency", "Rust_in_Maize": "common_rust",
    "Stem_Borer": "stem_borer", "Trips": "thrips", "Zinc_Deficiency_in_Maize": "zinc_deficiency",
}


def load_manifest() -> List[Dict]:
    with MANIFEST.open() as handle:
        return list(csv.DictReader(handle))


@torch.no_grad()
def extract(rows: List[Dict]) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    if FEATURES.exists():
        data = np.load(FEATURES, allow_pickle=True)
        cache = dict(zip(data["keys"].tolist(), data["feats"]))
    todo = [r["path"] for r in rows if f"{r['path']}|0" not in cache]
    if todo:
        processor = AutoImageProcessor.from_pretrained(VISION_ID)
        model = AutoModel.from_pretrained(VISION_ID, torch_dtype=torch.float32).eval()
        for start in range(0, len(todo), 16):
            batch_paths = todo[start:start + 16]
            images, keys = [], []
            for path in batch_paths:
                image = Image.open(ROOT / path).convert("RGB")
                images += [image, ImageOps.mirror(image)]
                keys += [f"{path}|0", f"{path}|1"]
            inputs = processor(images=images, return_tensors="pt")
            out = model.vision_model(inputs["pixel_values"]).last_hidden_state
            feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=-1)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            for key, vector in zip(keys, feat.numpy()):
                cache[key] = vector.astype(np.float32)
            print(f"  features {min(start + 16, len(todo))}/{len(todo)}", end="\r")
        FEATURES.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FEATURES, keys=np.array(list(cache.keys()), dtype=object), feats=np.stack(list(cache.values())))
        print()
    return cache


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def nll(t):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(labels)), labels].mean()
    return float(minimize_scalar(nll, bounds=(0.2, 10), method="bounded").x)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-images", type=int, default=10)
    parser.add_argument("--C", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_manifest()
    counts = Counter(r["label"] for r in rows)
    classes = sorted(c for c, n in counts.items() if n >= args.min_images)
    rows = [r for r in rows if r["label"] in classes]
    print("classes:", {c: counts[c] for c in classes})
    print("skipped (too few images):", {c: n for c, n in counts.items() if c not in classes})
    cache = extract(rows)
    index = {c: i for i, c in enumerate(classes)}

    # ---- held-out split by group (agronomist images always train) ----------
    eval_pool = [r for r in rows if r["source"] != "agronomist"]
    y_pool = np.array([index[r["label"]] for r in eval_pool])
    groups = np.array([r["group"] for r in eval_pool])
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    _, test_idx = next(splitter.split(np.zeros(len(eval_pool)), y_pool, groups))
    test_paths = {eval_pool[i]["path"] for i in test_idx}
    train_rows = [r for r in rows if r["path"] not in test_paths]
    test_rows = [r for r in rows if r["path"] in test_paths]

    def matrix(selected: List[Dict], augment: bool):
        X, y = [], []
        for r in selected:
            for flip in ((0, 1) if augment else (0,)):
                X.append(cache[f"{r['path']}|{flip}"])
                y.append(index[r["label"]])
        return np.stack(X), np.array(y)

    X_train, y_train = matrix(train_rows, augment=True)
    X_test, y_test = matrix(test_rows, augment=False)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=args.C, max_iter=4000, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)
    logits = clf.decision_function(scaler.transform(X_test))
    temperature = fit_temperature(logits, y_test)
    probs = softmax(logits / temperature)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    report = {"classes": classes, "temperature": temperature, "n_train": len(train_rows), "n_test": len(test_rows),
              "per_class": {}, "per_source": {}}
    per_class = defaultdict(lambda: [0, 0, 0])
    for truth, p, t3 in zip(y_test, pred, top3):
        stats = per_class[classes[truth]]
        stats[0] += 1
        stats[1] += int(p == truth)
        stats[2] += int(truth in t3)
    for name, (n, top1, in_top3) in sorted(per_class.items()):
        report["per_class"][name] = {"n": n, "top1": round(top1 / n, 3), "top3": round(in_top3 / n, 3)}
    report["top1"] = round(float((pred == y_test).mean()), 3)
    report["top3"] = round(float(np.mean([t in row for t, row in zip(y_test, top3)])), 3)
    healthy = index.get("healthy")
    report["sick_called_healthy"] = int(sum(1 for t, p in zip(y_test, pred) if t != healthy and p == healthy))

    # ---- current production head on the same held-out images ---------------
    old_clf = joblib.load(ROOT / "maize_vlm_classifier.joblib")
    old_scaler = joblib.load(ROOT / "maize_vlm_scaler.joblib")
    old_names = [OLD_NAME_MAP.get(n, n) for n in joblib.load(ROOT / "maize_class_names.joblib")]
    old_probs = old_clf.predict_proba(old_scaler.transform(X_test))
    old_pred = [old_names[i] if old_probs[k, i] * 100 >= 60 else "healthy"
                for k, i in enumerate(old_probs.argmax(1))]
    old_top3 = [[old_names[i] for i in np.argsort(-row)[:3]] for row in old_probs]
    truth_names = [classes[t] for t in y_test]
    report["baseline_old_head"] = {
        "top1": round(float(np.mean([p == t for p, t in zip(old_pred, truth_names)])), 3),
        "top3": round(float(np.mean([t in t3 for t, t3 in zip(truth_names, old_top3)])), 3),
        "sick_called_healthy": int(sum(1 for p, t in zip(old_pred, truth_names) if p == "healthy" and t != "healthy")),
    }
    print(json.dumps(report, indent=2))

    # ---- save the train-split head + held-out list for the pipeline eval ----
    EVAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, EVAL_ARTIFACTS / "classifier.joblib")
    joblib.dump(scaler, EVAL_ARTIFACTS / "scaler.joblib")
    train_counts = Counter(r["label"] for r in train_rows)
    (EVAL_ARTIFACTS / "meta.json").write_text(json.dumps({
        "classes": classes, "temperature": temperature, "support": {c: train_counts[c] for c in classes},
        "vision_id": VISION_ID, "note": "trained on the train split only; use with test_split.json",
    }, indent=2))
    (EVAL_ARTIFACTS / "test_split.json").write_text(json.dumps(
        [{"path": r["path"], "label": r["label"], "source": r["source"]} for r in test_rows], indent=1))

    # ---- refit on all data and save -----------------------------------------
    X_all, y_all = matrix(rows, augment=True)
    scaler = StandardScaler().fit(X_all)
    clf = LogisticRegression(C=args.C, max_iter=4000, class_weight="balanced").fit(scaler.transform(X_all), y_all)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ARTIFACTS / "classifier.joblib")
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    meta = {
        "classes": classes,
        "temperature": temperature,
        "support": {c: counts[c] for c in classes},
        "vision_id": VISION_ID,
        "eval": report,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {ARTIFACTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
