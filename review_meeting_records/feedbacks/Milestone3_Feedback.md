## Feedback for Milestone 3

- **No per-class detection evaluation** - Report per-class precision/recall and confusion matrix to identify weak categories.
- **No detection failure analysis** - Include qualitative examples of false positives/negatives and discuss impact on navigation.
- **No real-world detection validation** - Test model on live camera/video data under realistic conditions. We did see a small demo of it but not explicitly planned in the document.
- **Depth evaluation on very small sample size** - Increase evaluation to larger dataset + real-world samples.
- **No validation of depth with detected objects** - Measure distance error at bounding box locations (object-level depth accuracy).
- **No temporal consistency in depth** - Add temporal smoothing (EMA/filtering across frames) to avoid flicker.
- **No real-world depth validation** - Test depth estimation on live indoor scenes beyond NYU dataset.
- **Hardcoded thresholds (3m, 1.5m) not justified** - Tune thresholds using empirical experiments and validation metrics.
- **No depth failure analysis** - Analyze failure cases (glass, reflective surfaces, low light) with examples.
- **No quantitative evaluation of navigation decisions** - Create a labeled dataset (scene -> correct action) and measure accuracy/collision rate.
- **No ground truth for decision logic** - Define expected actions for test scenarios to validate correctness.
- **Heuristic parameters (zone split, risk formula) not justified** - Perform relevant studies to validate design choices.
- **No handling of multi-object conflict scenarios** - Define rules/tests for conflicting risks (left vs right obstacles).
- **No temporal planning (only frame-wise decisions)** - Introduce short-term memory or trajectory smoothing.
- **No failure handling in decision layer** - Add fallback logic when detection/depth is uncertain.
- **SLM-based logic not implemented or evaluated** - Either implement and compare quantitatively or remove from scope.
- **System performs obstacle avoidance, not full navigation** - Clearly define scope or extend toward goal-directed navigation logic.
- **TTS not implemented (only planned)** - Implement at least one baseline TTS model and demonstrate output.
- **No TTS model selection** - Benchmark candidates and choose one with justification.
- **No integration of TTS with pipeline** - Connect decision output -> TTS -> audio and test end-to-end.
- **No latency measurement for TTS** - Measure command-to-audio delay and Real-Time Factor (RTF).
- **No intelligibility evaluation** - Conduct basic human evaluation for clarity of commands.
- **Multilingual support undecided** - Finalize language scope and evaluate selected languages.
- **No edge deployment validation for TTS** - Test on target hardware (CPU/edge device).
- **No true end-to-end pipeline validation** - Run full pipeline on continuous video stream.
- **No end-to-end latency measurement** - Measure camera -> detection -> depth -> decision -> audio delay.
- **No deployment validation on edge device** - Test complete system on target hardware (e.g., Raspberry Pi).
- **No robustness testing** - Evaluate under low light, clutter, occlusion, and dynamic environments.
- **No system-level metrics** - Define and report:
  - navigation success rate
  - collision avoidance rate
  - response time
- **No cross-module failure handling** - Design fallback strategies when one module fails (e.g., rely more on detection if depth fails).

At this stage there cannot be any design choice that are yet TBD, for example TTS model or Decision algorithm. Please choose one way or the other with proper justification.
