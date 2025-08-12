# GPU-Based Quantum Circuit Simulator

This project is a high-performance, GPU-accelerated quantum circuit simulator. It models the evolution of multi-qubit quantum states by applying quantum gates directly on the GPU using CUDA.

The motivation behind this simulator was to learn more about GPU programming as well as quantum computing. 
## How to use 
### 1. build and run
``` 
git clone git@github.com:semzit/PsiGPU.  #clone 
cd PsiGPU
mkdir build #create build directory
cd build
cmake .. # build 
make
./PsiGPU # run executable
```
### 2. Example usage 

## State Vector 
The state vector is represented as an array of  cuDoubleComplex which allows the amplitude to represent both its real and imaginary parts

Amplitude: $a+bi$  

The complete state vector for a system of two qubits would be represented by:

$\alpha|00> + \beta|01> + \gamma|10> + \delta|11>$  

Or

$(a+bi)\ |00> + (c+di)\ |01> + (e+fi)\ |10> + (g+hi)\ |11>$  
## Capabilities
-
-
-
## Kernel Fusion 
Because an iterative approach when applying qauntum gates leads to incorrect results,  gates applied to non disjoint qubits must be fused together
## Circuit gates
### Kernel example
```cpp
__global__ void hadamard_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit){
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = gridDim.x * blockDim.x; 

    size_t stride = 1ULL << target_qubit; // calculates the second number in the pair because amplitudes are stored in a 2d array

    for (size_t current_pair_idx = global_idx; current_pair_idx < (1ULL << (num_qubits - 1)); current_pair_idx += grid_size){
        // Calculate the base index for this pair
        size_t base_idx = (current_pair_idx / stride) * (stride * 2) + (current_pair_idx % stride);

        size_t idx0 = base_idx;
        size_t idx1 = base_idx | stride;

        cuDoubleComplex amp0 = state_vector[idx0];
        cuDoubleComplex amp1 = state_vector[idx1];
   
        const double inv_sqrt2 = 1.0 / sqrt(2.0);

        state_vector[idx0] = make_cuDoubleComplex(cuCadd(amp0, amp1).x * inv_sqrt2, cuCadd(amp0, amp1).y * inv_sqrt2);
        state_vector[idx1] = make_cuDoubleComplex(cuCsub(amp0, amp1).x * inv_sqrt2, cuCsub(amp0, amp1).y * inv_sqrt2);
    }
}
```
### Pauli Gates
### Phase Gates
### Rotation Gates
### Hadamard Gate

### CNOT Gate
### SWAP Gate
### Toffoli Gate

## Probability calulation 

Because the amplitude is given by the $a+bi$ the maginitude of probabilty for the system to resolve to that basis state can be given by $|a+bi|^2$ 

## Long-Term Vision & Goals
- Lightweight, fast, GPU powered quantum simulator 
- Alternative to high-performance GPU-based quantum emulation
- Contribute to quantum-itlassical hybrid simulation methods


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

