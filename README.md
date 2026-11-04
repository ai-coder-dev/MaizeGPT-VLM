# 🌽 MaizeGPT VLM: Maize Pathology Diagnostics

> **A lightweight, edge-ready Hybrid Vision-Language Model (VLM) for real-time maize (*Zea mays*) pathology diagnosis and grounded agronomic advisory.**

---

### 📊 Performance Summary
| Metric | Score | Split Size |
| :--- | :---: | :--- |
| **Holdout Test Accuracy** | **100.00%** | 200 images (10% holdout) |
| **Validation Accuracy** | **98.99%** | 398 images (20% split) |
| **Training Set** | — | 1,394 images (70% split) |
| **Total Dataset** | — | 1,992 balanced images |

---

## 1. Dataset & Classes

The system detects **12 target maize disorders**:

* **Fungal & Bacterial Diseases:** Charcoal Rot, Grey Leaf Spot, Maydis Leaf Blight, Northern Leaf Blight, Common Rust.
* **Insect Pests:** Fall Armyworm, Stem Borer, Thrips.
* **Nutrient Deficiencies:** Nitrogen (N), Phosphorus (P), Potassium (K), Zinc (Zn).

---

## 2. Model Architecture

Instead of monolithic 8B VLMs that cause out-of-memory errors on standard tiers, this pipeline uses a **two-stage hybrid architecture**:

```text
       [Maize Leaf Image]
               │
               ▼
   [Stage 1: Google SigLIP Base Patch16 (384px)]
               │
   (Spatial mean + max token pooling)
               ▼
   [Calibrated Multi-Class Diagnostic Head]
               │
   (Disease prediction + confidence score)
               ▼
   [Stage 2: Qwen2.5-0.5B-Instruct + Agronomic Grounding]
               │
               ▼
   [Field Prescription: Chemical Dosages, Bio-Controls, 14-Day Plan]
```

### Why This Hybrid Setup?
* **Google SigLIP Base (Vision):** Provides dense visual token embeddings for detecting both localized lesions and broad chlorotic patterns.
* **Qwen2.5-0.5B-Instruct (Language):** Compact reasoning model (~500M parameters) that operates smoothly on low-memory environments.
* **Agronomic Grounding:** Enforces strict maize-specific pathology protocols to eliminate crop-hallucination errors.
