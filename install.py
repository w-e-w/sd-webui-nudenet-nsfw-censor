import launch

if not launch.is_installed('onnxruntime') and not launch.is_installed('onnxruntime-gpu'):
    import torch.cuda as cuda
    if cuda.is_available():
        launch.run_pip('install onnxruntime-gpu')
    else:
        launch.run_pip('install onnxruntime')
