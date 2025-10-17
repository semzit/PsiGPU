# GPU-Based Quantum Circuit Simulator
This project is a high-performance, GPU-accelerated quantum circuit simulator. It models the evolution of multi-qubit quantum states by applying quantum gates directly on the GPU using CUDA.

The motivation behind this simulator was to learn more about GPU programming as well as quantum computing. 

## Table of Contents
- [Capabilities](https://github.com/semzit/PsiGPU/?tab=readme-ov-file#capabilities)
- [Build from source](https://github.com/semzit/PsiGPU/?tab=readme-ov-file#how-to-use)
-   [State Vector](https://github.com/semzit/PsiGPU/?tab=readme-ov-file#state-vector)
- [Kernel Fusion](https://github.com/semzit/PsiGPU/?tab=readme-ov-file#kernel-fusion)
- [Resources](https://github.com/semzit/PsiGPU/?tab=readme-ov-file#resources)
## Capabilities 

- [ ] Enable multi qubit gates

- [ ] Probability visualizer 

- [x] Kernel fusion 

- [x]  Kernels to enable the use of single qubit gates

- [x] Circuit visualizer

- [x]  Simple ui

## How to use 
### 1. Build and run
``` 
git clone git@github.com:semzit/PsiGPU.  #clone 
cd PsiGPU
mkdir build #create build directory
cd build
cmake .. # build 
make
./PsiGPU # run executable
``` 

## State Vector 
The state vector is represented as an array of  cuDoubleComplex which allows the amplitude to represent both its real and imaginary parts

Amplitude: $a+bi$  

The complete state vector for a system of two qubits would be represented by:

$\alpha|00> + \beta|01> + \gamma|10> + \delta|11>$  

Or

$(a+bi)\ |00> + (c+di)\ |01> + (e+fi)\ |10> + (g+hi)\ |11>$  
## Kernel Fusion 
Because an iterative approach when applying quantum gates and launching individual gpu kernels every time is expensive this project works by iterating throught the gates within the kernel. 

## GPU Kernel
 ```cpp
 __global__ void applyGate(cuDoubleComplex* stateVec, const Gate* gates, int numQubits, int numGates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Thread index 
    int dim = 1 << numQubits;  // Total amplitude count
    if (idx >= dim) return;

    // Iterate through gates
    for (int gate = 0; gate < numGates; gate++){  
        int i = gates[gate].targets[0];   // Get control qubit
        int pairIdx = idx ^ (1 << i);  // Index of second amplitude in the pain (the beta to a given alpha)

        if(idx < pairIdx){
            cuDoubleComplex a = stateVec[idx]; 
            cuDoubleComplex b = stateVec[pairIdx]; 

            // Multiply and add (matrix-vector multiplication)
            stateVec[idx] =     cuCadd(cuCmul(gates[gate].matrix[0], a), 
                                       cuCmul(gates[gate].matrix[1], b)); 
            
            stateVec[pairIdx] = cuCadd(cuCmul(gates[gate].matrix[2], a), 
                                       cuCmul(gates[gate].matrix[3], b)); 
        }
        __syncthreads(); // Dont continue until all threads are done
    }
}
 ```

## Probability calulation 

Because in quantum mechanics the probability is the square of the amplitude, We can sqaure $a+bi$ to get the probabilty that the system will resolve to that basis state: $|a+bi|^2$ 

## Long-Term Vision & Goals
- Lightweight, fast, GPU powered quantum simulator 
- Alternative for high-performance GPU-based quantum emulation


## Resources 
CUDA/GPU programming: 
- https://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/SLIDES/
- https://www.learncpp.com/
- https://google.github.io/googletest/primer.html 

Quantum Computing: 
- https://quantum.country/
- https://youtu.be/tsbCSkvHhMo?si=DjhDKAKPLa0PlkgT
- https://youtu.be/RQWpF2Gb-gU?si=qExsQVy-IvSGIXRE
## Tech Stack
- Language: C++17, CUDA 11
- Testing: GoogleTest
- GPU: NVIDIA 4060

##  License
MIT 

