# GPU-Based Quantum Circuit Simulator

This project is a high-performance, GPU-accelerated quantum circuit simulator. It models the evolution of multi-qubit quantum states by applying quantum gates directly on the GPU using CUDA.

The motivation behind this simulator was to learn more about GPU programming as well as quantum computing. 



## Long-Term Vision & Goals
- Lightweight, fast, GPU powered quantum simulator 
- Alternative to high-performance GPU-based quantum emulation
- Contribute to quantum-classical hybrid simulation methods



###  Phase 1: Core Simulator Foundation 
- [] State vector representation using GPU memory
- [] Basic single-qubit gates: `X`, `Y`, `Z`, `H`, `S`, `T`
- [] Two-qubit `CNOT` gate support
- [] Gate application via custom CUDA kernels
- [] Simple CLI for hardcoded/test circuits

## Resources 

CUDA/GPU programming - https://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/SLIDES/

## Tech Stack
- Language: C++17
- GPU: NVIDIA



##  License
MIT 

