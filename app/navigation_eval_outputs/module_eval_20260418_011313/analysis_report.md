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
- yolo: mean=40.43296500167344, p50=22.282099991571158, p95=33.445610018679844
- depth: mean=107.09707833254167, p50=99.66335001809057, p95=125.30356501520146
- hazard_scan: mean=9.926593334724506, p50=9.82020000810735, p95=14.719480011262924
- navigation: mean=0.03498666774248704, p50=0.03130000550299883, p95=0.06196997564984485
- tts: mean=0.004161666826500247, p50=0.003600012860260904, p95=0.006445017061196259
- pipeline: mean=159.4996033333397, p50=132.1546999970451, p95=179.34002998517823

## Navigation + TTS
- Command distribution: {'Turn right.': 14, 'Go straight. The closest object is a chair, about 1.5 meters away.': 1, 'Go straight. The closest object is a chair, about 2.1 meters away.': 1, 'Go straight. The closest object is a sofa, about 1.7 meters away.': 1, 'Turn left': 8, 'The way ahead is clear. You can keep going straight.': 5, 'Go straight. The closest object is a table, about 1.2 meters away.': 1, 'Go straight. The closest object is a table, about 1.5 meters away.': 1, 'Go straight. The closest object is a sofa, about 1.5 meters away.': 1, 'Searching for path. Turn back.': 1, 'Go straight. The closest object is a refrigerator, about 2.3 meters away.': 1, 'Go straight. The closest object is a person, about 1.8 meters away.': 1, 'Go straight. The closest object is a bed, about 1.7 meters away.': 1, 'Go straight. The closest object is a chair, about 1.7 meters away.': 1, 'Go straight. The closest object is a potted plant, about 2.4 meters away.': 1, 'Go straight. The closest object is a chair, about 1.3 meters away.': 3, 'Go straight. The closest object is a chair, about 1.6 meters away.': 2, 'Go straight. The closest object is a refrigerator, about 1.6 meters away.': 1, 'Go straight. The closest object is a refrigerator, about 0.7 meters away.': 1, 'Obstacle ahead. Move slightly right.': 1, 'Go straight. The closest object is a person, about 1.3 meters away.': 1, 'Go straight. The closest object is a person, about 7.7 meters away.': 1, 'Go straight. The closest object is a chair, about 2.0 meters away.': 2, 'Go straight. The closest object is a person, about 2.0 meters away.': 1, 'Go straight. The closest object is a potted plant, about 1.6 meters away.': 1, 'Go straight. The closest object is a refrigerator, about 1.5 meters away.': 1, 'Go straight. The closest object is a table, about 2.6 meters away.': 1, 'Go straight. The closest object is a table, about 2.7 meters away.': 1, 'Go straight. The closest object is a table, about 2.4 meters away.': 1, 'Go straight. The closest object is a chair, about 2.8 meters away.': 1, 'Go straight. The closest object is a chair, about 1.9 meters away.': 1, 'Go straight. The closest object is a refrigerator, about 3.1 meters away.': 1}
- TTS generated count: 60
- TTS failed count: 0
