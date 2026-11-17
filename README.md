# MaizeGPT VLM: maize disease, pest and deficiency diagnosis

Streamlit app that diagnoses maize problems from farmer photos. It covers 32 problems named as in ICAR-IIMR publications, and gives ICAR-IIMR management advice only when it is confident.

## Results (v2, 17 Nov 2026)

Measured on 226 held-out images. The evaluated classifier never saw them, and photos are grouped by source so the same plant or contributor is never in both training and test. The images come from open datasets and ICAR-IIMR bulletin photos.

| | Old model | v2 pipeline |
|---|---:|---:|
| Insect pests: top answer correct | 35% | 82% (97% in top 3) |
| Diseases: correct cause in top 3 | 43% | 96% |
| Sick plant called "healthy" | 44 / 201 | 0 / 201 |
| Accuracy when the app says "confident" | n/a | 91% |

The old model's "100% holdout accuracy" came from augmented copies of the same 166 training photos appearing in both train and test, so it did not reflect field performance.

Re-run the evaluation with `python evals/run_image_eval.py --be-env <path to .env with OPENAI_API_KEY>`.

## How it works

```text
farmer photo (leaf, stalk, cob, tassel or insect)
  -> SigLIP base 384px features, test-time augmentation (original / flip / centre crop)
  -> calibrated classifier head: 19 trained classes incl. healthy (models/iimr_head_v2)
  -> SigLIP zero-shot plant-part check
  -> second check by gpt-6-luna vision, constrained to all 32 IIMR problems
     (runs when the head is unsure, the class is confusable or under-trained, or the photo is not a leaf)
  -> fused top-3 causes + status:
       confident  -> diagnosis + ICAR-IIMR management (chemicals allowed)
       likely     -> field checks, non-chemical steps, "confirm with KVK before spraying"
       uncertain  -> ask for a clearer photo; possible causes listed
       retake     -> not a usable maize photo
```

**Safety rules**
- "Healthy" is shown only when both models agree. Otherwise the app says "No clear problem visible (not confirmed)".
- Chemicals are shown only for confident diagnoses. They come from `knowledge/advice.json`, which is exported from the MaizeGPT ICAR-IIMR table. Pesticides banned in India are never shown.
- The 13 problems with too few photos to train are covered by the second check only: BLSB, stalk rots, polysora rust, head smut, ear rot and others.

## Files

| Path | What |
|---|---|
| `app.py`, `ui.py`, `assets/` | Streamlit app |
| `vlm_pipeline.py` | `MaizeDiagnoser`: classifier, part check, second check, fusion, advice |
| `knowledge/iimr_problems.json` | the 32 IIMR problems with field signs and look-alikes |
| `knowledge/advice.json` | ICAR-IIMR management per problem (built by `training/export_advice.py`) |
| `models/iimr_head_v2/` | deployed head (all data); `models/iimr_head_v2_eval/` is train-split only, used for evaluation |
| `training/` | dataset builder, IIMR PDF photo extraction and quality filter, head training, advice export |
| `evals/run_image_eval.py` | held-out evaluation |

The training images (`data/`, ~2,700 images, gitignored) and their provenance (`data/manifest.csv`, plus rejected photos in `data/rejected/` with reasons) are rebuilt with:

```bash
python training/build_dataset.py --agronomist-dir "<MaizeGPT-BE>/test_data"
python training/extract_pdf_figures.py --raw-manifest "<MaizeGPT-BE>/knowledge/raw/manifest.json"
python training/filter_iimr_photos.py --be-env "<MaizeGPT-BE>/.env"
python training/train_head.py
python training/export_advice.py --table "<MaizeGPT-BE>/knowledge/tables/pests_diseases.json"
```

Image sources: RootsGoods agronomist photos; Mendeley *Corn Leaf Disease Classification Dataset* (CC BY 4.0); University of Pretoria *Diseases of maize in the field* (CC BY 4.0); iNaturalist research-grade CC0 / CC BY / CC BY-SA photos; and photos from ICAR-IIMR bulletins.

## Deployment

Add `OPENAI_API_KEY` to the Streamlit secrets to enable the second check. Without it the app falls back to the SigLIP classifier alone and is more cautious. Optional: `SECOND_CHECK_MODEL` (default `gpt-6-luna`).
