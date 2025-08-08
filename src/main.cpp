#include <iostream>
#include <string>
#include "quantum_state.h"
#include "circuit.h"
#include "circuit_executer.h"
#include "ui.h"


int main(){
    std::cout << "Welcome to PsiGPU!" << "\n"; 
    Circuit circuit = Circuit(5);  // create circuit
    CircuitExecuter executer; 
    QuantumState q_state = QuantumState(5); 


    while (true){
        std::string input ; 
        std::cin >> input ; 
        input = cleanString(input);
        
        if(input == "help"){
            printHelp(); 
        }else if (input == "print circuit"){
            circuit.printCircuitCLI(); 
        }else if (input == "run"){
            executer.execute(circuit, q_state); 
        }
    }
     
}