# Module-wise Evaluation Report

## YOLO Object Detection
- Precision: 0.5027624309392266
- Recall: 0.3063973063973064
- F1: 0.38075313807531386
- Mean IoU: 0.8468144683690841

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
- yolo: mean=30.567773333314108, p50=12.346099998467253, p95=14.213290001680427
- depth: mean=89.76665500000915, p50=92.7552499997546, p95=99.42646999825227
- hazard_scan: mean=7.9453066663215095, p50=7.945150000523427, p95=11.283055001331377
- navigation: mean=0.03070333326225712, p50=0.030500001230393536, p95=0.04865999671892494
- tts: mean=24.939496666532552, p50=0.41984999916167, p95=23.208474999590848
- pipeline: mean=266.25858499998384, p50=236.2021499993716, p95=339.257840000936

## Navigation + TTS
- Command distribution: {'Clear path on your right. Turn right.': 14, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.1 meters away.': 3, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.0 meters away.': 3, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 1.7 meters away.': 2, 'Clear path on your left. Turn left.': 9, 'The way ahead is clear. You can keep going straight.': 5, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 3.1 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 2.0 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.7 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.7 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a potted plant, about 2.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 1.6 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.6 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 0.7 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.5 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 1.3 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a cabinet, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.6 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.1 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a cabinet, about 2.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 3.0 meters away.': 1}
- TTS generated count: 60
- TTS failed count: 0
