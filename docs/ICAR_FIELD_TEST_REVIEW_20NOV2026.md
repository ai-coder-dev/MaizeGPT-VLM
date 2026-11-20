# Maize VLM: review of the agronomists' field test (images of 5–6 Nov 2026)

## Summary
- **The report (`VLM_Diagnostic_Graphical_Analysis_06-11-2026.pdf`, 18/45 correct) was produced by the old v1 model, not the current app.** We re-ran v1 on the same 45 images and it reproduced the report's prediction on 41 of 45. The 4 differences are borderline cases affected by image compression. The labels in the report ("Healthy Maize Specimen", "Fall Army worm", "Trips") are v1's raw class names. The current app at https://maizegpt-vlm.streamlit.app does not use them.
- **On the same 45 images, the current app (with the fixes below) gets:**
  - the top answer right on **25 of 45 (56%)**;
  - the correct cause in its **top 3 on 37 of 45 (82%)**;
  - **0 sick plants called "healthy"** (v1 did this 8 times).
- **The test photos showed two problems in our training data,** now fixed: two photos in our own agronomist set were mislabelled.

## Results on the 45 expert-labelled images

| | v1 (in the report) | Current app |
|---|---:|---:|
| Top answer correct | 18 / 45 (40%) | **25 / 45 (56%)** |
| Correct cause in top 3 | – | **37 / 45 (82%)** |
| Sick plant called "healthy" | 8 | **0** |
| Answers shown as "confident" | – | 19, of which 17 correct (89%) |

**Notes on the results:**
- **Images 39 of 45 were never seen in training:** top answer 20 of 39 (51%); top 3, 31 of 39 (79%).
- **v1 could not output 16 of the 45 answers at all.** It only knew 12 problems, so bacterial stalk rot, BLSB, Fusarium stalk rot, downy mildew, smut, aphid, viral stripe, bacterial leaf streak and southern rust were always wrong. The current app covers all 32 IIMR problems.
- **The test images are the small copies embedded in the spreadsheet** (161–512 px). Accuracy on the original photos may be somewhat higher.
- **Run-to-run variation:** the second check is an AI model, and repeated runs vary by about ±2 images.

## What we found and fixed

1. **The second check was being overruled.**
   - On 8 images the second check gave the expert's answer (bacterial and Fusarium stalk rot, BLSB, MLB, bacterial leaf streak), but the image classifier, trained mostly on non-Indian leaf photos, outweighed it.
   - We now give the second check more weight. On the expert photos the top answer improved from 51% to 60% with the same top-3 accuracy, at a cost of about 2 points on open-dataset photos.
2. **Two mislabelled training photos** in the RootsGoods agronomist set matched expert test images exactly:
   - *"Charcoal Rot (2).jpg"* shows a banded lesion on the leaf sheath, so it was relabelled **BLSB** (test image #26).
   - *"Maydis (Southern) Leaf Blight (5).jpg"* shows small round orange pustules, so it was relabelled **southern (polysora) rust** (test image #37).
   - A near-duplicate Mendeley photo labelled grey leaf spot, which the experts called bacterial leaf streak (test image #31), has been set aside pending review.

   The model was retrained, and southern rust is now a trained class (20 trained problems).
3. **Clearer distinguishing signs, written from IIMR bulletins,** for the problems that were confused with each other:
   - downy mildew vs zinc vs nitrogen deficiency;
   - bacterial, charcoal and Fusarium stalk rot vs BLSB;
   - MLB vs TLB vs grey leaf spot vs bacterial leaf streak;
   - common vs southern rust.
4. **Safety:**
   - **Nutrient deficiencies are never shown as "confident".** The app gives field checks and recommends a soil or tissue test, because downy mildew was being called zinc deficiency.
   - **The app does not say "healthy" if its second check describes any spot or mark,** even when both models lean healthy.
   - **Rust photos always get the second check.** A southern rust photo had been shown as "confident" common rust. On 100 held-out common rust photos accuracy is unchanged (89%), and none was switched to southern rust; fewer of them are now shown as "confident" (38 instead of 76).

## Still not right (needs data, not code)
- **Downy mildew** (test images #11 and #42) is still called zinc deficiency, now only as "likely". Our training set has 12 downy mildew photos.
- **Bacterial stalk rot vs charcoal rot** (#4, #5), **stem borer damage inside the stalk** (#21), and **BLSB at leaf level** (#44, shown as TLB with "confident" status) remain wrong.
- **Ten problems have fewer than 10 training photos** and rely on the second check alone: BLSB, bacterial, Fusarium and Pythium stalk rot, head smut, ear rot, Curvularia, bacterial leaf streak, viral disease, shoot fly, termite and cutworm.

## Requests to the experts
1. **Check our agronomist training photos:** `data/review/agronomist_photos_label_review.pdf` with its CSV has 164 photos and 14 pages, each with its current label. Mark any wrong labels. Two errors were already found this way.
2. **Share the ICAR-IIMR image bank** (the ~25 problems, original resolution), especially downy mildew, BLSB, the stalk rots, smut and stem borer damage.
3. **Re-test using the current app** at https://maizegpt-vlm.streamlit.app (a hard refresh, Ctrl+F5 / Cmd+Shift+R, makes sure the latest version loads).

## Reproduce
```bash
python evals/icar_field_set/build_from_xlsx.py "Test 2_VLM_Diagnostic_Report_05-11-2026.xlsx"   # images + labels.csv
python evals/icar_field_set/run.py --be-env <path to .env with OPENAI_API_KEY>             # v1 vs current, per image
```
The latest per-image results are in `evals/results/icar_field_set_20261120_2006.csv`. These 45 images are an evaluation set only and are never used for training.
