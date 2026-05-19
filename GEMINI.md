# Block Sparse Attention

## Project Overview

**Block Sparse Attention** is a Python library providing specialized sparse attention kernels for Large Language Models (LLMs). It is designed to optimize performance and reduce computational and memory bandwidth costs during inference, especially for long prompts. 

The library supports a dual-backend architecture:
1. **NVIDIA Backend:** Built as a PyTorch C++/CUDA extension heavily based on [FlashAttention](https://github.com/Dao-AILab/flash-attention) and CUTLASS.
2. **AMD Backend (ROCm):** Fully powered by a custom high-performance Triton engine, providing seamless acceleration on AMD GPUs (e.g., GFX1100/RDNA3) without any C++ compilation hurdles.

It supports assigning different attention patterns for different heads:
1.  **Dense Attention:** Full attention matrix.
2.  **Streaming Attention (Token Granularity):** Fixed number of sink and local tokens (like StreamingLLM).
3.  **Streaming Attention (Block Granularity):** Fixed number of sink and local blocks (block size = 128).
4.  **Blocksparse Attention:** Uses a custom block mask (block size = 128).

**Core Technologies:** Python, C++ (CUDA), PyTorch, CUTLASS, Triton (for AMD ROCm backend).

## Building and Installation

### Requirements
*   **OS:** Linux (Windows is partially referenced in `setup.py` but the README explicitly requires Linux)
*   **NVIDIA Hardware:** Ampere, Ada, or Hopper GPUs (with CUDA 11.6+ and PyTorch 1.12+)
*   **AMD Hardware:** ROCm-supported GPUs (e.g., RDNA3 / gfx1100 / Radeon RX 7900 XTX) with ROCm 6.0+ and PyTorch ROCm version.
*   **PyTorch:** 1.12 and above

### Installation

1.  **Install build dependencies:**
    ```bash
    pip install packaging ninja triton
    ```
2.  **Install the package:**
    ```bash
    python setup.py install
    ```
    *Note:*
    * *On NVIDIA platforms, the `setup.py` script attempts to download pre-built wheels from GitHub releases by default. If a wheel is not found, or if you set `BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE`, it will compile the CUDA extension from source.*
    * *On AMD ROCm platforms, `setup.py` automatically detects the environment, bypasses all CUDA-specific C++ extensions compilation, and transparently routes all operators to the pure Triton high-performance backend, ensuring a seamless, compile-free installation.*

## Testing and Benchmarking

Tests are located in the `block_sparse_tests/` directory and use `pytest`.

### Correctness Tests

Install pytest first: `pip install pytest`

*   **Forward Pass Only:**
    ```bash
    cd ./block_sparse_tests/fwd/test_correctness
    pytest full_test.py
    ```
*   **Forward and Backward Pass:**
    ```bash
    cd ./block_sparse_tests/fwd_bwd/test_correctness
    pytest full_test.py
    ```

### Performance Tests

Performance tests are standalone Python scripts.

*   **Forward Pass Only:**
    ```bash
    cd ./block_sparse_tests/fwd/test_performance/
    python token_streaming.py
    python blocksparse.py
    ```
*   **Forward and Backward Pass:**
    ```bash
    cd ./block_sparse_tests/fwd_bwd/test_performance/
    python block_streaming.py
    python blocksparse.py
    ```

## Development Conventions & Structure

*   **`block_sparse_attn/`**: Contains the Python interface and API wrappers (e.g., `block_sparse_attn_func`).
*   **`csrc/`**: Contains the C++ and CUDA source code for the kernels.
*   **`csrc/cutlass/`**: CUTLASS is used as a git submodule. `setup.py` will attempt to initialize it if missing.
*   **Environment Variables for Build:**
    *   `BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE`: Force a fresh local build from source instead of downloading wheels.
    *   `BLOCK_SPARSE_ATTN_SKIP_CUDA_BUILD=TRUE`: Skip CUDA compilation (useful for sdist).
    *   `BLOCK_SPARSE_ATTN_CUDA_ARCHS`: Semicolon-separated list of target architectures (e.g., `80;90`).
    *   `MAX_JOBS`: Controls parallel compilation jobs for Ninja.