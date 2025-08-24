#include <gtest/gtest.h>
#include "circuit_executer.h"
#include "circuit.h"

// Custom comparison helper for cuDoubleComplex
::testing::AssertionResult ComplexNear(const cuDoubleComplex& a,
                                       const cuDoubleComplex& b,
                                       double tol = 1e-9) {
    if (fabs(a.x - b.x) < tol && fabs(a.y - b.y) < tol) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "Expected (" << b.x << "," << b.y << ") but got ("
        << a.x << "," << a.y << ")";
}

TEST(NotTest, BasicAssertions) {
    int qubit_count = 1; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count);     
    CircuitExecuter executer; 

    circuit.addPualiX(0); 

    executer.execute(circuit, q_state); 

    std::vector<cuDoubleComplex> expected = {make_cuDoubleComplex(0,0), make_cuDoubleComplex(1,0)}; 

    auto actual = q_state.getStateVector();
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_TRUE(ComplexNear(actual[i], expected[i]));
    }
} 

TEST(PauliYTest, BasicAssertions){
    int qubit_count = 1; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count);     
    CircuitExecuter executer; 

    circuit.addPauliY(0); 

    executer.execute(circuit, q_state); 

    std::vector<cuDoubleComplex> expected = {make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,1)}; 

    auto actual = q_state.getStateVector();
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_TRUE(ComplexNear(actual[i], expected[i]));
    }

}

TEST(PauliZTest, BasicAssertions){
    int qubit_count = 1; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count);     
    CircuitExecuter executer; 

    circuit.addPauliZ(0); 

    executer.execute(circuit, q_state); 

    std::vector<cuDoubleComplex> expected = {make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0)}; 

    auto actual = q_state.getStateVector();
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_TRUE(ComplexNear(actual[i], expected[i]));
    }

}

TEST(PhaseSTest, BasicAssertions){
    int qubit_count = 1; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count);     
    CircuitExecuter executer; 


    circuit.addHadamard(0); // first simulate superposition
    circuit.addPhaseS(0); 

    executer.execute(circuit, q_state); 

    std::vector<cuDoubleComplex> expected = {make_cuDoubleComplex(M_SQRT1_2,0), make_cuDoubleComplex(0,M_SQRT1_2)}; 

    auto actual = q_state.getStateVector();
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_TRUE(ComplexNear(actual[i], expected[i]));
    }

}

TEST(PhaseTTest, BasicAssertions){
    int qubit_count = 1; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count);     
    CircuitExecuter executer; 
    
    circuit.addHadamard(0); // first simulate superposition
    circuit.addPhaseT(0); 

    executer.execute(circuit, q_state); 

    std::vector<cuDoubleComplex> expected = {make_cuDoubleComplex(M_SQRT1_2,0), make_cuDoubleComplex(0.5,0.5)}; 

    auto actual = q_state.getStateVector();
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_TRUE(ComplexNear(actual[i], expected[i]));
    }

}

TEST(HadamardTest, BasicAssertions){
    int qubit_count = 1; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count);     
    CircuitExecuter executer; 

    circuit.addHadamard(0); 

    executer.execute(circuit, q_state); 

    std::vector<cuDoubleComplex> expected = {make_cuDoubleComplex(M_SQRT1_2,0), make_cuDoubleComplex(M_SQRT1_2,0)}; 

    auto actual = q_state.getStateVector();
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_TRUE(ComplexNear(actual[i], expected[i]));
    }

}