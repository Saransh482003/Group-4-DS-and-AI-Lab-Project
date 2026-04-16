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
- yolo: mean=22.156683332286775, p50=12.793049972970039, p95=14.83298991806804
- depth: mean=105.9937483301231, p50=97.98120002960786, p95=146.7621800198685
- hazard_scan: mean=7.972026678423087, p50=7.855000032577664, p95=11.288550042081624
- navigation: mean=0.03291499063683053, p50=0.030049995984882116, p95=0.05877498188056051
- tts: mean=638.842669990845, p50=707.6582000008784, p95=782.2009400581009
- pipeline: mean=891.485024994472, p50=863.3680999628268, p95=1071.5595399262384

## Navigation + TTS
- Command distribution: {'Clear path on your right. Turn right.': 14, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.1 meters away.': 3, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.0 meters away.': 3, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 1.7 meters away.': 2, 'Clear path on your left. Turn left.': 9, 'The way ahead is clear. You can keep going straight.': 5, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 3.1 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 2.0 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a sofa, about 1.7 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.7 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a potted plant, about 2.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 1.6 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.6 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 0.7 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 1.5 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a person, about 1.3 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a cabinet, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a refrigerator, about 1.5 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.6 meters away.': 2, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.4 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a chair, about 2.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 2.1 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a cabinet, about 2.8 meters away.': 1, 'The way ahead is clear. Keep moving straight. The closest object is a table, about 3.0 meters away.': 1}
- TTS generated count: 60
- TTS failed count: 0
