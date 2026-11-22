# PathSafe-Ellipse :: network training & test report

checkpoint: `ellipse_net_best.pt` (epoch 20)

## Held-out test split (unseen map groups)

| metric | value |
|---|---|
| test samples | 33490 |
| composite loss | 0.4080 |
| centre offset error | 1.205 m |
| semi-axis a error | 1.089 m |
| semi-axis b error | 0.617 m |
| orientation error (angle_valid samples) | 15.96 deg |
| ellipse IoU (patch raster) | 0.7684 |
| path point inside prediction (rho <= 0.9) | 0.9673 |
| predicted ellipse collision-free | 0.7202 (4000 checked) |
| **safe AND contains path point** | **0.6915** |

## Validation split

| metric | value |
|---|---|
| val samples | 31567 |
| composite loss | 0.4428 |
| centre offset error | 1.314 m |
| ellipse IoU | 0.7401 |
| safe AND contains path point | 0.6697 |

预测可视化：`reports/test_predictions.png`；训练曲线：`reports/training_curves.png`。

