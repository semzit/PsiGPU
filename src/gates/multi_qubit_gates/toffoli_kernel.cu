#include "circuit_executer.h"
#include <cuComplex.h>



__global__ void toffoli_kernel(cuDoubleComplex* state_vector, int num_qubits, int target_qubit, int control_qubit, int second_control_qubit){}