#include <gtest/gtest.h>
#include "circuit_executer.h"
#include "circuit.h"


TEST(HelloTest, BasicAssertions) {
    int qubit_count = 5; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count);     
    CircuitExecuter executer; 

    circuit.addHadamard(1); 

    executer.execute(circuit, q_state); 

} 