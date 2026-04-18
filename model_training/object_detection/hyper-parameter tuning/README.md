# Hyperparameter Tuning

This directory contains hyperparameter tuning runs for YOLO object detection. Each Jupyter notebook corresponds to a specific tuning experiment, mapped below:

## `dsai-unified-dataset-run-1.ipynb`
- **Project / Tuning Type:** `mosaic_mixup`

**Parameter Grid:**
```python
[
    {"mosaic": 0.5, "mixup": 0.0},
    {"mosaic": 1.0, "mixup": 0.0},
    {"mosaic": 1.0, "mixup": 0.2},
    {"mosaic": 1.0, "mixup": 0.3},
    {"mosaic": 0.5, "mixup": 0.2},
]
```

---

## `dsai-unified-dataset-run-2.ipynb`
- **Project / Tuning Type:** `scale_tuning`

**Parameter Grid:**
```python
[
    {"scale":0.3, "translate":0.10, "mosaic":1.0, "mixup":0.0},
    {"scale":0.5, "translate":0.10, "mosaic":1.0, "mixup":0.0},
    {"scale":0.7, "translate":0.10, "mosaic":1.0, "mixup":0.0},
    {"scale":0.5, "translate":0.15, "mosaic":1.0, "mixup":0.0},
    {"scale":0.5, "translate":0.20, "mosaic":1.0, "mixup":0.2},
]
```

---

## `dsai-unified-dataset-run-3.ipynb`
- **Project / Tuning Type:** `image_size`

**Parameter Grid:**
```python
[
    {"imgsz": 640},
    {"imgsz": 768},
    {"imgsz": 832},
    {"imgsz": 960},
    {"imgsz": 768, "scale": 0.7},
]
```

---

## `dsai-unified-dataset-run-4.ipynb`
- **Project / Tuning Type:** `batch_size`

**Parameter Grid:**
```python
[
    {"batch": 16},
    {"batch": 32},
    {"batch": 48},
    {"batch": 64},
    {"batch": 32, "imgsz": 832},
]
```

---

## `dsai-unified-dataset-run-5.ipynb`
- **Project / Tuning Type:** `light_aug`

**Parameter Grid:**
```python
[
    {"hsv_h": 0.01, "hsv_s": 0.5, "hsv_v": 0.3},
    {"hsv_h": 0.015, "hsv_s": 0.6, "hsv_v": 0.4},
    {"hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.5},
    {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.3},
    {"hsv_h": 0.02, "hsv_s": 0.5, "hsv_v": 0.5},
]
```

---

## `dsai-unified-dataset-run-6.ipynb`
- **Project / Tuning Type:** `rotation_perspective`

**Parameter Grid:**
```python
[
    {"degrees": 5, "perspective": 0.0},
    {"degrees": 10, "perspective": 0.0005},
    {"degrees": 15, "perspective": 0.001},
    {"degrees": 20, "perspective": 0.001},
    {"degrees": 10, "perspective": 0.002},
]
```

---

## `dsai-unified-dataset-run-7.ipynb`
- **Project / Tuning Type:** `copypaste_erasing`

**Parameter Grid:**
```python
[
    {"copy_paste": 0.0, "erasing": 0.2},
    {"copy_paste": 0.1, "erasing": 0.3},
    {"copy_paste": 0.2, "erasing": 0.4},
    {"copy_paste": 0.3, "erasing": 0.4},
    {"copy_paste": 0.1, "erasing": 0.5},
]
```

---

## `dsai-unified-dataset-run-8.ipynb`
- **Project / Tuning Type:** `optimizer_lr`

**Parameter Grid:**
```python
[
    {"optimizer": "SGD", "lr0": 0.01},
    {"optimizer": "SGD", "lr0": 0.005},
    {"optimizer": "AdamW", "lr0": 0.001},
    {"optimizer": "AdamW", "lr0": 0.0005},
    {"optimizer": "AdamW", "lr0": 0.0003},
]
```

---

## `dsai-unified-dataset-run-9.ipynb`
- **Project / Tuning Type:** `weight_decay`

**Parameter Grid:**
```python
[
    {"weight_decay": 0.0005},
    {"weight_decay": 0.001},
    {"weight_decay": 0.0001},
    {"weight_decay": 0.00005},
    {"weight_decay": 0.0005, "optimizer": "AdamW"},
]
```

---

## `dsai-unified-dataset-run-10.ipynb`
- **Project / Tuning Type:** `final_combo`

**Parameter Grid:**
```python
[
    {"imgsz": 768, "batch": 32, "scale": 0.5, "mosaic": 1.0, "mixup": 0.2},
    {"imgsz": 832, "batch": 32, "scale": 0.5, "mosaic": 1.0, "mixup": 0.2},
    {"imgsz": 832, "batch": 48, "scale": 0.5, "mosaic": 1.0, "mixup": 0.2},
    {"imgsz": 768, "batch": 32, "scale": 0.7, "mosaic": 1.0, "mixup": 0.3},
    {"imgsz": 832, "batch": 32, "scale": 0.7, "mosaic": 1.0, "mixup": 0.3},
]
```

---

## `dsai-unified-dataset-run-11.ipynb`
- **Project / Tuning Type:** `lr_warmup`

**Parameter Grid:**
```python
[
    {"lr0": 0.01,  "lrf": 0.01, "warmup_epochs": 3},
    {"lr0": 0.005, "lrf": 0.01, "warmup_epochs": 3},
    {"lr0": 0.003, "lrf": 0.01, "warmup_epochs": 5},
    {"lr0": 0.001, "lrf": 0.01, "warmup_epochs": 5},
    {"lr0": 0.001, "lrf": 0.001, "warmup_epochs": 5},
]
```

---

## `dsai-unified-dataset-run-12.ipynb`
- **Project / Tuning Type:** `loss_gain`

**Parameter Grid:**
```python
[
    {"box": 7.5, "cls": 0.5, "dfl": 1.5},
    {"box": 10.0, "cls": 0.5, "dfl": 1.5},
    {"box": 12.0, "cls": 0.5, "dfl": 1.5},
    {"box": 10.0, "cls": 0.7, "dfl": 1.5},
    {"box": 10.0, "cls": 0.5, "dfl": 2.0},
]
```

---

