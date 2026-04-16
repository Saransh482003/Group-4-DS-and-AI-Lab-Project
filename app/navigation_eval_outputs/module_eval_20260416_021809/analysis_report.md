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

## Latency (ms)
- yolo: mean=29.422799990667652, p50=12.594200030434877, p95=14.943734992993996
- depth: mean=103.38695000003402, p50=98.14799996092916, p95=135.77730995602903
- navigation: mean=0.02789832845640679, p50=0.027349975425750017, p95=0.04349498194642364
- tts: mean=741.5640533358479, p50=725.6168000167236, p95=835.2413199958391
- pipeline: mean=875.7197366658753, p50=836.8397999438457, p95=965.2321999310516

## Navigation + TTS
- Command distribution: {'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.1 meters away.': 3, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.0 meters away.': 5, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 1.7 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.7 meters away.': 4, 'The way ahead is clear. You can keep going straight.': 6, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 3.1 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 1.5 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.4 meters away.': 1, 'Clear path on your right. Turn right.': 5, 'Clear path on your left. Turn left.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.8 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 2.0 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.5 meters away.': 5, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.7 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a potted plant, about 2.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 1.6 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.6 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 0.7 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 1.6 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a bed, about 0.9 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 1.3 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a cabinet, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a bed, about 1.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.6 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.9 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 1.3 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.1 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a cabinet, about 2.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 3.0 meters away.': 1}
- TTS generated count: 60
- TTS failed count: 0
