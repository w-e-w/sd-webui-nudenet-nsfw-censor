import launch

if not launch.is_installed('onnxruntime') and not launch.is_installed('onnxruntime-gpu'):
    import torch.cuda as cuda
    if cuda.is_available():
        def get_onnxruntime_extra_index():
            """
            onnxruntime-gpu requires wheel from a different index for CUDA 12
            https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x
            https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/onnxruntime-cuda-12/PyPI/onnxruntime-gpu/overview
            https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/onnxruntime-cuda-12/connect
            """
            import subprocess
            import re
            try:
                if re.search(r'CUDA\s+Version:\s+([0-9.]+)\s*', subprocess.check_output(["nvidia-smi"]).decode()).group(1).startswith('12'):
                    return ' --extra-index-url "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"'
            except Exception as e:
                print(f'Unable to get CUDA version: {e}')
            return ''

        launch.run_pip(f'install onnxruntime-gpu{get_onnxruntime_extra_index()}', 'onnxruntime-gpu')
    else:
        launch.run_pip('install onnxruntime', 'onnxruntime')
