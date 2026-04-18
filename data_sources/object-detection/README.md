# Data Sources - Object Detection

This directory contains the data preparation, inspection, and preprocessing pipelines used to construct the final unified dataset for training the YOLO object detection model. It includes scripts and documentation for acquiring various public datasets, extracting specific classes of interest, and normalizing annotations into a standard format.

## Files Description

### COCO Dataset
*   **`coco_dataset_preprocessing.ipynb`**: General pipeline for selectively downloading and preprocessing specific classes of interest from the COCO dataset.
*   **`coco-bed.ipynb`**: Notebook specifically tasked with extracting, processing, and validating instances of the "bed" class from COCO text/image files to bolster underrepresented classes.
*   **`coco-refregerator.ipynb`**: Notebook tasked with extracting and processing instances of the "refrigerator" class from the COCO dataset.

### HomeObjects-3K
*   **`homeobject-3k_data_preprocessing.ipynb`**: Data transformation script that normalizes the HomeObjects-3K dataset into our unified YOLO bounding box format.

### SUN RGB-D
*   **`sunrgbd_data_inspection.ipynb`**: Exploratory Data Analysis (EDA) uncovering the class distribution, bounding box statistics, and general structure of the SUN RGB-D dataset.
*   **`sunrgbd-preprocessing.md`**: Text documentation capturing the methodology and instructions required to preprocess the SUN RGB-D files effectively.

### General & Unified
*   **`dataset_links.md`**: A reference file containing the original URLs, sources, and citation links for every dataset leveraged in this project.
*   **`indoor_object_detection.ipynb`**: General, exploratory, or baseline notebook covering initial experiments on indoor-specific object datasets.
*   **`dsai_unified_data_inspection.ipynb`**: Final inspection and EDA notebook. Analyzes the aggregated, synthesized `dsai-unified-dataset` after merging all data sources above, ensuring that class maps and bounding boxes are correctly unified before training begins.
