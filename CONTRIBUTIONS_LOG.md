# Project Contribution Log

This document tracks individual contributions made by team members across project milestones.

---

## Milestone 1

### Problem Definition & Literature Review
- **Saransh Saini** – Preparation and drafting of the Milestone-1 document, including problem formulation, system scope definition, literature review of assistive navigation systems, and outlining the proposed system architecture and evaluation framework. Collaborated on making the Review Meet Milestone 1 PPT.
- **Samyuktha Shriram** – Contributed to analysis of existing approaches in assistive navigation systems, with particular focus on the depth estimation component and considerations for integrating depth information into navigation decision making. Collaborated on making the Review Meet Milestone 1 PPT.
- **Rohit Prajapat** – Contributed to system ideation and evaluation planning by discussing real-world deployment considerations, user safety requirements, and possible metrics for assessing navigation performance. Also developed a quick Proof-of-Concept mockup demo on Colab Notebook for visualizing project flow.
- **Divyang Panchasara** – Participated in problem definition discussions and helped refine the project scope by identifying key system modules (object detection, depth estimation, navigation reasoning) and their role in assistive navigation. Gave implementation suggestions based on practical experience with CV models.
- **Prasoon Shukla** – Contributed to literature review discussions and system design planning for Milestone-1 and helped identify limitations in current object detection and proximity-based methods during the literature review phase.

---

## Milestone 2

###  Dataset Preparation
- **Saransh Saini** – Amalgamation of the three modules' preprocessing, dataset info submitted and preparation of Milestone-2 document. Worked alongwith Prasoon for object detection module. Collaborated on making the Review Meet Milestone 2 PPT.
- **Samyuktha Shriram** – Dataset analysis for depth estimation module, identification of RGB-D datasets (NYU Depth V2, ScanNet, SUN RGB-D, DIODE, Matterport3D), study of dataset characteristics and preprocessing requirements. Collaborated on making the Review Meet Milestone 2 PPT.
- **Rohit Prajapat** – Exploration of depth estimation datasets and repository setup for dataset sources, documentation of dataset links, and preparation for integration of monocular depth estimation (MiDaS) into the project pipeline. Worked alongwith Samyuktha in the depth estimation module.
- **Divyang Panchasara** – Identified and collected suitable datasets for the TTS and Lightweight LLM module.
- **Prasoon Shukla** - Identified and collected suitable dataset for object detection model

---

## Milestone 3

###  Dataset Preparation
- **Saransh Saini** – Successfully integrated all standalone modules—object detection, monocular depth estimation, and basic decision logic—into a cohesive, functional raw pipeline. Focused extensively on profiling and optimizing subsystem inference times to resolve critical initial latency bottlenecks. By streamlining data flow between these modules, ensured the overall system maintained the strict real-time responsiveness and usability required for practical, real-world assistive navigation scenarios.
- **Samyuktha Shriram** – Conducted comparative benchmarking of multiple depth estimation models (MiDaS Small in particular) on the NYU Depth V2 dataset using ground truth depth maps. Analyzed performance using standard metrics and visual comparisons, contributing to the selection of Depth Anything V2 (Indoor Metric Small) as the final model. Prepared depth module slides and contributed to the milestone report documentation.
- **Rohit Prajapat** – Actively collaborated in interpreting benchmarking results and discussing trade-offs between model accuracy and efficiency for practical deployment. Supported the team in refining evaluation strategies and ensuring consistency in metric computation.
- **Divyang Panchasara** – Conducted comparative analysis of multiple TTS engines with a focus on real-time performance characteristics such as Real-Time Factor (RTF), synthesis latency, and responsiveness. Evaluated suitability of different engines for assistive navigation use cases and contributed to defining initial evaluation metrics and benchmarking approach.
- **Prasoon Shukla** - Explored and validated object detection architectures by implementing YOLOv8n for initial experimentation and inference on Google Colab. Contributed to identification and selection of relevant object classes from collected datasets for the navigation use case. Participated in evaluating preprocessing requirements to improve model input quality.

---

## Milestone 4

###  Dataset Preparation
- **Saransh Saini** – Conducted an iterative evaluation of multiple YOLO architectures to identify the most efficient object detector for the pipeline. Led extensive hyperparameter tuning experiments, training and validating 60 distinct model configurations across 12 separate Kaggle compute sessions. Through this rigorous comparative analysis, successfully trained, finalized, and exported the optimal model weights, balancing high precision with minimal inference delay for deployment.
- **Samyuktha Shriram** – Focused on improving depth prediction stability through post-processing techniques, including temporal smoothing using EMA and scale alignment methods (direct inference, median scaling, scale-shift alignment). Conducted qualitative failure analysis across challenging scenarios (reflections, motion, low light) via live-cam images captured and evaluated the impact of smoothing on prediction consistency. Prepared depth module slides and contributed to the milestone report.
- **Rohit Prajapat** – Assisted in experimenting with different smoothing parameters and validating their impact on temporal consistency across varied scenarios. Contributed to discussions on balancing stability improvements with responsiveness in dynamic environments.
- **Divyang Panchasara** – Designed and implemented evaluation pipelines for measuring TTS performance, including RTF and latency across diverse datasets. Developed multiple execution modes (sequential and parallel) for different pipeline configurations (detection, depth, and combined), enabling structured benchmarking of system performance.
- **Prasoon Shukla** - Investigated and applied image preprocessing techniques to enhance detection performance, particularly under low-visibility conditions. Proposed and implemented a preprocessing decision mechanism (boolean function) to dynamically apply enhancements such as CLAHE (Contrast Limited Adaptive Histogram Equalization) when required.

---

## Milestone 5

###  Dataset Preparation
- **Saransh Saini** – Integrated newly refined models into the pipeline and conducted comprehensive, end-to-end performance checks. To rigorously validate the system, hand-picked a diverse subset of 60 challenging indoor images from the SUNRGBD dataset. Engineered a custom preprocessing script to extract 2D bounding boxes from complex 3D annotations, compiling a robust validation dataset. Subsequently executed and documented system-wide evaluations against these challenging scenes.
- **Samyuktha Shriram** – Evaluated the depth module through real-world testing and structured failure case analysis, including multiple live scenarios. Contributed to identifying limitations of object-dependent navigation and proposed the use of depth-based hazard zones as an improvement. Participated in system-level evaluation, interpretation of metrics, and documentation. Prepared depth module slides and contributed to the milestone report.
- **Rohit Prajapat** – Helped in organizing and documenting real-world test observations, ensuring clear mapping between failure cases and potential system improvements. Participated in brainstorming sessions to refine navigation strategies based on depth insights.
- **Divyang Panchasara** – Defined and applied comprehensive evaluation metrics at both module and system level, including FPS, component-wise latency, navigation command distribution, and safety indicators such as center-blocked rate. Contributed to system-level performance analysis, including comparison of sequential vs parallel execution and assessment of TTS overhead in the full pipeline. Integrated real-world datasets (EgoBlind) for evaluation.
- **Prasoon Shukla** - Contributed to evaluation and analysis of the object detection module by compiling training and inference results into structured reports. Participated in system-level testing through smoke tests on recorded video inputs, validating detection performance and pipeline behavior across real-world scenarios.

---

## Milestone 6

###  Dataset Preparation
- **Saransh Saini** – Engineered and integrated the depth-based hazard zones algorithm, significantly improving blind navigation safety by prioritizing purely geometric depth data. Drastically reduced audio feedback latency by implementing an in-memory LRU phrase cache for Text-to-Speech commands. Engineered the Streamlit web UI and successfully deployed the live pipeline to Hugging Face Spaces. Authored cross-platform Docker and auto-setup scripts to guarantee seamless code reproducibility.
- **Samyuktha Shriram** – Contributed to final system integration and documentation, including articulation of depth module behavior, hazard zone logic, and system-level evaluation in the final report. Assisted in ideation of deployment-ready components and aided in testing-review of the Streamlit web app made. Contributed to presentation preparation and final documentation.
- **Rohit Prajapat** – Supported final integration efforts by reviewing module interactions and ensuring clarity in system-level documentation. Contributed to polishing presentation materials and verifying that key ideas were communicated effectively.
- **Divyang Panchasara** – Contributed to final system integration and documentation, including development of orchestration pipelines for end-to-end benchmarking across multiple configurations. Supported deployment readiness through frame processing optimizations (sampling, striding, skipping) for efficient real-time execution across images, videos, and live streams. Contributed to final report and presentation preparation.
- **Prasoon Shukla** - Supported deployment and final integration by testing the complete pipeline on local hardware (Ubuntu system with NVIDIA GPU), verifying end-to-end functionality including preprocessing, inference, and output generation. Contributed to preparation of presentation slides and documentation summarizing methodology and results.
