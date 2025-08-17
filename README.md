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

## State Vector 
The state vector is represented as an array of  cuDoubleComplex which allows the amplitude to represent both its real and imaginary parts

Amplitude: $a+bi$  

The complete state vector for a system of two qubits would be represented by:

$\alpha|00> + \beta|01> + \gamma|10> + \delta|11>$  

Or

$(a+bi)\ |00> + (c+di)\ |01> + (e+fi)\ |10> + (g+hi)\ |11>$  
## Capabilities
- Simulate a qauntum state 
- Apply qauntum gates that act on a single qubit
## Kernel Fusion 
Because an iterative approach when applying qauntum gates and launching invidaul gpu kernels every time is expensive this project works by putting them all together through kernel fusion. 

- If gates act on overlapping qubits then you muliply the matrices before applying: $U_{fused} = U_B \ \cdot \ U_A$
- If gates act on different qubits you can take the tensor product between them: $ U_A \ \otimes \ U_b$
## Circuit gates

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

