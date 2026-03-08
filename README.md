# Action-Oriented Indoor Navigation Assistance for the Visually Impaired

## Overview

Visually impaired individuals navigating indoor environments often rely
on assistive technologies that announce detected objects. However,
object identification alone does not indicate whether an obstacle lies
directly in the user's walking path or requires an immediate change in
movement.

For example, hearing *"chair detected"* does not inform the user whether
they are about to collide with it or can safely continue forward.

This project focuses on **action-oriented indoor navigation
assistance**, where visual perception is converted into **movement
guidance rather than descriptive awareness**.

Instead of only reporting objects, the system provides **directional
instructions such as "move left" or "move right" when obstacles pose a
collision risk.**

------------------------------------------------------------------------

# Problem Statement

Existing assistive systems typically perform **object detection and
verbal announcements**, but they do not:

-   Identify whether the object lies **within the user's walking path**
-   Estimate **collision risk**
-   Provide **immediate movement guidance**

As a result, visually impaired users may still face difficulty
determining how to safely navigate around obstacles.

------------------------------------------------------------------------

# Proposed Solution

The system performs **real-time perception and rule-based reasoning** to
convert visual inputs into **actionable navigation instructions**.

The solution focuses specifically on two indoor navigation risks:

1.  **Obstacle Collision**
2.  **Blocked Walking Path**

within a **near-field range of 0--3 meters**.

The system uses:

-   **Webcam input** (proof-of-concept)
-   **Real-time object detection**
-   **Monocular depth estimation**
-   **Rule-based spatial reasoning**

to generate **concise navigation prompts**.

------------------------------------------------------------------------

# System Architecture

Webcam Input\
↓\
Object Detection Model\
↓\
Monocular Depth Estimation\
↓\
Spatial Risk Assessment\
↓\
Rule-Based Reasoning Engine\
↓\
Navigation Guidance (Audio Output)

------------------------------------------------------------------------

# Key Concepts

### Walking Zone

The **walking zone** represents the **central region of the camera
frame**, corresponding to the user's forward movement path.

### Collision Risk

An obstacle is considered a **collision risk** when:

-   It lies **within the walking zone**
-   Its **distance falls below a predefined threshold**

A **collision** is defined as **any physical contact between the
participant and an obstacle during navigation.**

### Directional Guidance

If the walking zone is blocked, the system evaluates **lateral zones**
to determine a safer path and generates instructions such as:

-   **Move Left**
-   **Move Right**

------------------------------------------------------------------------

# Reasoning Framework

The navigation decisions are generated using an **explicit rule-based
reasoning system**.

Rules rely on:

-   Object distance from the user
-   Spatial overlap with the walking zone
-   Lateral clearance in adjacent zones

This approach ensures:

-   **Interpretability**
-   **Deterministic behavior**
-   **Feasibility within the project timeline**

------------------------------------------------------------------------

# Baseline Comparison

To evaluate the effectiveness of action-oriented navigation guidance,
the proposed system will be compared with a **baseline system**.

  System     Output
  ---------- ----------------------------------------------
  Baseline   Object labels only (e.g., "Chair detected")
  Proposed   Directional instructions (e.g., "Move Left")

Both systems will use the **same detection and depth estimation models**
to ensure fair comparison.

------------------------------------------------------------------------

# Experimental Setup

Evaluation will be conducted in a **controlled indoor obstacle course**.

### Environment

-   Indoor path length: **10 meters**
-   **5 standardized obstacles**

### Participants

-   **Minimum 5 blindfolded participants**
-   Blindfolding ensures **consistent testing conditions**

### Trials

-   **3 trials per system mode per participant**
-   Total: **15+ navigation trials**

Multiple trials help reduce **learning effects and randomness**.

------------------------------------------------------------------------

# Evaluation Metrics

System performance will be evaluated using the following metrics:

-   **Collision Count** -- Number of physical contacts with obstacles.
-   **Navigation Completion Time** -- Total time required to complete
    the obstacle course.
-   **Corrective Stops** -- A stop is counted when the participant halts
    ≥3 seconds, steps backward, or requires manual intervention.
-   **Reaction Time** -- Time between audio instruction delivery and
    user response movement.

------------------------------------------------------------------------

# Latency and Usability Requirements

End-to-end latency is measured from:

Obstacle enters walking zone → System processes scene → Navigation
instruction delivered

The system must satisfy two usability criteria:

1.  Users should **react smoothly without abrupt stopping**
2.  Users should **walk continuously without frequent pauses caused by
    delayed feedback**

------------------------------------------------------------------------

# Implementation Constraints

-   **Offline execution**
-   **Webcam-based input**
-   **Audio feedback via earphones**
-   Indoor navigation only
-   Near-field obstacle detection (0--3 meters)

------------------------------------------------------------------------

# Project Objective

The objective of this project is to evaluate whether **translating
visual perception into actionable navigation guidance** can improve
**safe indoor mobility for visually impaired individuals** compared to
traditional object announcement systems.

------------------------------------------------------------------------

# Repository Structure

```
project-root
│
├── app
│   └── app.py                     # Main application script
│
├── data
│   └── dataset_links.md           # Dataset references and links
│
├── docs
│   └── problem-statement.pdf      # Project problem statement
│
├── feedbacks
│   ├── Milestone1.md              # Feedback for milestone 1
│   └── Milestone2.md              # Feedback for milestone 2
│
├── reports
│   ├── Milestone1.pdf             # Milestone 1 report
│   └── Milestone2.pdf             # Milestone 2 report
│
├── CHANGELOG.md                   # Contribution log of team members
│
└── README.md                      # Project documentation
```
# Team Details

**Group Number:** 4\
**Client:** Mr. M Nagarajan\
**Number of Members:** 5

| Name               | Email                                                                   | GitHub           |
| ------------------ | ----------------------------------------------------------------------- | ---------------- |
| Saransh Saini      | [22f1001123@ds.study.iitm.ac.in](mailto:22f1001123@ds.study.iitm.ac.in) | Saransh482003    |
| Divyang Panchasara | [22f1000411@ds.study.iitm.ac.in](mailto:22f1000411@ds.study.iitm.ac.in) | 22f1000411       |
| Samyuktha Shriram  | [22f2001444@ds.study.iitm.ac.in](mailto:22f2001444@ds.study.iitm.ac.in) | SamyukthaSh24    |
| Prasoon Shukla     | [23f3003434@ds.study.iitm.ac.in](mailto:23f3003434@ds.study.iitm.ac.in) | 23f3003434       |
| Rohit Prajapat     | [22f1001536@ds.study.iitm.ac.in](mailto:22f1001536@ds.study.iitm.ac.in) | rohitblpprajapat |
