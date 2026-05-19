# Prevent library conflicts in ROCm environments by preloading BOTH correct hsa-runtime and HIP early before any PyTorch imports
import os
import ctypes
import subprocess

try:
    # 尝试通过 hipconfig 寻找真实 ROCm 根目录并双重预加载
    rocm_path = subprocess.check_output(['hipconfig', '-R'], universal_newlines=True).strip()
    if 'opt/rocm' in rocm_path:
        parts = rocm_path.split(os.sep)
        for part in parts:
            if part.startswith('rocm-'):
                real_rocm = os.path.join('/opt', part)
                hsa_path = None
                for lib_name in ['libhsa-runtime64.so.1', 'libhsa-runtime64.so']:
                    p = os.path.join(real_rocm, 'lib', lib_name)
                    if os.path.exists(p):
                        hsa_path = p
                        break
                
                hip_path = None
                for lib_name in ['libamdhip64.so.7', 'libamdhip64.so']:
                    p = os.path.join(real_rocm, 'lib', lib_name)
                    if os.path.exists(p):
                        hip_path = p
                        break
                
                if hsa_path and hip_path:
                    try:
                        ctypes.CDLL(hsa_path, mode=ctypes.RTLD_GLOBAL)
                        ctypes.CDLL(hip_path, mode=ctypes.RTLD_GLOBAL)
                        print(f"conftest early preload SUCCESS! loaded: {hsa_path} and {hip_path}")
                    except Exception as e:
                        print(f"conftest early preload FAILED: {e}")
except Exception as e:
    print(f"conftest error: {e}")
