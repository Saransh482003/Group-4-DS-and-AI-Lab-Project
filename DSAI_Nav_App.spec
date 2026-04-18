# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['E:\\dsai_group4_project\\app\\app.py'],
    pathex=['E:\\dsai_group4_project\\Depth-Anything-V2', 'E:\\dsai_group4_project\\app'],
    binaries=[],
    datas=[('E:\\dsai_group4_project\\model_training\\object_detection\\best-weights\\YOLO11s-Final-Training.pt', 'model_weights'), ('E:\\dsai_group4_project\\model_training\\depth_estimation\\model_weights\\depth_anything_v2_metric_hypersim_vits.pth', 'model_weights'), ('E:\\dsai_group4_project\\app\\piper', 'piper'), ('E:\\dsai_group4_project\\app\\piper_voices', 'piper_voices'), ('E:\\dsai_group4_project\\Depth-Anything-V2', 'Depth-Anything-V2')],
    hiddenimports=['ultralytics', 'torch', 'cv2', 'depth_anything_v2', 'mechanics'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DSAI_Nav_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DSAI_Nav_App',
)
