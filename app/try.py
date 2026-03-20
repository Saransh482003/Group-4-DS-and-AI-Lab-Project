# Least-squares scale+shift alignment (applied per-sample)
valid = gt > 0.1
A     = np.stack([pred[valid], np.ones_like(pred[valid])], axis=1)  # [N, 2]
scale, shift = np.linalg.lstsq(A, gt[valid], rcond=None)[0]
pred_aligned = np.clip(pred * scale + shift, 0.1, 20.0)
