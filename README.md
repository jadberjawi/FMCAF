# FMCAF
This is the modified version of the ultralytics Library allowing the replication of my workm which invlves the FMCAF module

# 🔧 Custom Ultralytics YOLO (Multimodal + Preprocessing Support)

This repository contains a **modified version of the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** library to enable:

- Direct loading of `.npy` files for multimodal input (e.g., RGB + IR).
- Insertion of custom preprocessing modules **before** the YOLO backbone.
- Clean modular handling of each modality (e.g., splitting after concatenation).

---

## 📁 Key Modifications

### 🧩 1. Input File Format Support (`.npy`)
- Modified the data loading pipeline to support **`.npy`** files instead of standard image formats.
- Each `.npy` file is expected to be a **concatenated tensor** of two modalities (e.g., RGB + IR).
- During preprocessing, the modalities are **split** and processed independently.

### 🧪 2. Custom Preprocessing Integration
- Introduced a hook into the model's `forward()` function to allow passing inputs through a **preprocessing module** **before** the YOLO backbone.
- You can plug in any custom logic for:
  - Frequency filtering
  - Normalization
  - Modality-specific enhancement
  - Cross-attention fusion, etc.

### 📦 3.  Module Mapping

| Original Name | Module Name in the paper     | Purpose / Functionality                                      |
|---------------|-------------------|---------------------------------------------------------------|
| `mefa`        | `MCAF`            | Multimodal Cross Attention Fusion for enhanced early fusion  |
| `rsr`         | `Freq-Filter`     | Redundant Spectrum Removal via frequency-based filtering     |

