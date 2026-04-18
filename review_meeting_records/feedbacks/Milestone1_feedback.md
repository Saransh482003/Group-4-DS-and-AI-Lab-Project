## Feedback for Milestone 1 
Recommended Changes

- The document defines appropriate evaluation metrics such as collision count, completion time, corrective stops, and reaction time. However, it does not specify quantitative success criteria for these metrics. The team should define what level of improvement over the baseline would be considered meaningful so that the evaluation leads to clear and objective conclusions.

- The proposal mentions the use of publicly available indoor datasets for training or fine-tuning models but does not identify specific datasets. The team should specify which datasets will be used, explain their relevance for indoor obstacle detection and depth estimation, and clarify whether any additional custom data collection will be required.

- The rule-based spatial reasoning module, which appears to be the core contribution of the project, is described only at a high level. The team should clearly explain how the walking zone is defined, how obstacle overlap with the walking path will be determined, and how decision thresholds will be chosen.

- Although monocular depth estimation is proposed as a key component, the document does not discuss its limitations or potential inaccuracies. The team should briefly acknowledge challenges such as scale ambiguity, lighting sensitivity, and possible depth estimation errors in indoor environments.

- Since the system supports safety-critical navigation, the proposal should describe how it handles uncertain or failure scenarios. For example, the team should clarify how the system responds to unreliable detections or conflicting signals, and whether a conservative fallback instruction such as “stop” will be used.

- The proposal would benefit from a brief description of the intended hardware setup and deployment assumptions. While a monocular RGB camera is mentioned, the document does not specify camera placement, processing device, or considerations related to real-time system performance.

- Add a chart to depict the process pipeline. Include and name a lightweight LLM in the pipeline for context aware text generation for Auditory processing using TTS.

- Add appropriate references to cite literature review at the end of the document, cross indexing where necessary.