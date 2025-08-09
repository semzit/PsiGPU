#include <iostream>
#include <string>
#include "quantum_state.h"
#include "circuit.h"
#include "circuit_executer.h"
#include <iostream>
#include <cctype>
#include <string>
#include <algorithm>

void printHelp(){
    std::cout << "Commands: \n";
    std::cout << "  addGate <gateName> <target_qubit> [<control_qubit>] [<second_control_qubit>]\n";
    std::cout << "  print - prints a view of current circtui\n";
    std::cout << "  run - executes current circuit and returns measurements\n";
}

std::string cleanString(std::string& s) {
    // Lowercase
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Remove whitespace
    s.erase(std::remove_if(s.begin(), s.end(),
                           [](unsigned char c) { return std::isspace(c); }),
            s.end());

    return s;
}

int main(){
    int qubit_count = -1;

    std::cout << "Welcome to PsiGPU!" << "\n";
    while (qubit_count > 20 || qubit_count< 1){
        std::cout << "Enter number of qubits you want: ";
        std::cin >> qubit_count;  
        if (std::cin.fail()) {
            std::cout << "Invalid input. Please enter a number.\n";
            std::cin.clear(); 
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }

    CircuitExecuter executer; 
    Circuit circuit = Circuit(qubit_count);  
    QuantumState q_state = QuantumState(qubit_count); 

    while (true){
        std::string input ;
        std::cout << "Command: ";  
        std::cin >> input ; 
        input = cleanString(input);
        
        if(input == "help"){
            printHelp(); 
        }else if (input == "print"){
            circuit.printCircuitCLI(); 
        }else if (input == "run"){
            executer.execute(circuit, q_state); 
            
        }else if (input == "quit"){
            return 0; 
        }
    }
     
}
