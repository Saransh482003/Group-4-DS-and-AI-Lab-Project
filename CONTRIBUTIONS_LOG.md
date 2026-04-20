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
- **Saransh Saini** – 
- **Samyuktha Shriram** – Conducted comparative benchmarking of multiple depth estimation models (MiDaS Small in particular) on the NYU Depth V2 dataset using ground truth depth maps. Analyzed performance using standard metrics and visual comparisons, contributing to the selection of Depth Anything V2 (Indoor Metric Small) as the final model. Prepared depth module slides and contributed to the milestone report documentation.
- **Rohit Prajapat** – 
- **Divyang Panchasara** – 
- **Prasoon Shukla** - 

---

## Milestone 4

###  Dataset Preparation
- **Saransh Saini** – 
- **Samyuktha Shriram** – Focused on improving depth prediction stability through post-processing techniques, including temporal smoothing using EMA and scale alignment methods (direct inference, median scaling, scale-shift alignment). Conducted qualitative failure analysis across challenging scenarios (reflections, motion, low light) via live-cam images captured and evaluated the impact of smoothing on prediction consistency. Prepared depth module slides and contributed to the milestone report.
- **Rohit Prajapat** – 
- **Divyang Panchasara** – 
- **Prasoon Shukla** - 

---

## Milestone 5

###  Dataset Preparation
- **Saransh Saini** – 
- **Samyuktha Shriram** – Evaluated the depth module through real-world testing and structured failure case analysis, including multiple live scenarios. Contributed to identifying limitations of object-dependent navigation and proposed the use of depth-based hazard zones as an improvement. Participated in system-level evaluation, interpretation of metrics, and documentation. Prepared depth module slides and contributed to the milestone report.
- **Rohit Prajapat** – 
- **Divyang Panchasara** – 
- **Prasoon Shukla** - 

---

## Milestone 6

###  Dataset Preparation
- **Saransh Saini** – 
- **Samyuktha Shriram** – Contributed to final system integration and documentation, including articulation of depth module behavior, hazard zone logic, and system-level evaluation in the final report. Assisted in ideation of deployment-ready components and aided in testing-review of the Streamlit web app made. Contributed to presentation preparation and final documentation.
- **Rohit Prajapat** – 
- **Divyang Panchasara** – 
- **Prasoon Shukla** - 
