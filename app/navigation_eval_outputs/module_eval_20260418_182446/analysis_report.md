# Module-wise Evaluation Report

## YOLO Object Detection
- Precision: 0.5816326530612245
- Recall: 0.3838383838383838
- F1: 0.46247464503042596
- Mean IoU: 0.8649772678979439

## Depth Estimation
- AbsRel mean: 0.5352248637382723
- RMSE (m) mean: 46.79271390433307
- Delta1 mean: 0.5167456808874444
- Valid ratio mean: 1.0

## Depth Hazard Detection
- Danger threshold (m): 1.2
- Warning threshold (m): 2.0
- Mean danger pixels: 27714.483333333334
- Mean warning pixels: 156398.66666666666
- Images with danger pixels: 44
- Mean nearest hazard depth (m): 1.0789586436950553

## Latency (ms)
- yolo: mean=29.33176166843623, p50=17.0926499995403, p95=20.60088000289397
- depth: mean=88.50518999824999, p50=88.6231000040425, p95=96.96396001963876
- hazard_scan: mean=8.924243333846485, p50=8.741450001252815, p95=13.678445003461093
- navigation: mean=0.027965001936536282, p50=0.025300018023699522, p95=0.06973000563448295
- tts: mean=0.00418000272475183, p50=0.0039000005926936865, p95=0.0059050158597528934
- pipeline: mean=128.35606000201855, p50=116.7561000038404, p95=128.97542998980498

## Navigation + TTS
- Command distribution: {'Turn right.': 14, 'Go straight.': 36, 'Turn left.': 8, 'Searching for path. Turn back.': 1, 'Move slightly right.': 1}
- TTS generated count: 60
- TTS failed count: 0
