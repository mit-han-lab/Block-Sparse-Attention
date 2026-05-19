# Prevent library conflicts in ROCm environments by preloading the correct hsa-runtime and HIP libraries early
try:
    import os
    import ctypes
    import subprocess
    rocm_path = subprocess.check_output(['hipconfig', '-R'], universal_newlines=True).strip()
    if 'opt/rocm' in rocm_path:
        parts = rocm_path.split(os.sep)
        for part in parts:
            if part.startswith('rocm-'):
                real_rocm = os.path.join('/opt', part)
                for lib_name in ['libhsa-runtime64.so.1', 'libhsa-runtime64.so']:
                    p = os.path.join(real_rocm, 'lib', lib_name)
                    if os.path.exists(p):
                        try:
                            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                            break
                        except Exception:
                            pass
except Exception:
    pass

# Fallback default paths
for p in ['/opt/rocm/lib/libhsa-runtime64.so.1', '/opt/rocm/lib/libhsa-runtime64.so']:
    try:
        import os
        import ctypes
        if os.path.exists(p):
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            break
    except Exception:
        pass

__version__ = "0.0.2"

from block_sparse_attn.block_sparse_attn_interface import (
    block_sparse_attn_func,
    token_streaming_attn_func,
    block_streaming_attn_func,
)

from . import utils