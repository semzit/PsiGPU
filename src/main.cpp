#include <iostream>
#include <string>
#include <sstream>
#include "quantum_state.h"
#include "circuit.h"
#include "circuit_executer.h"
#include <iostream>
#include <cctype>
#include <string>
#include <algorithm>


void printHelp(){
    std::cout << "Commands: \n";
    std::cout << "  add <gateName> <target_qubit> [<control_qubit>] [<second_control_qubit>]\n";
    std::cout << "  print - prints a view of current circuit\n";
    std::cout << "  run - executes current circuit and returns measurements\n";
    std::cout << "  gates - returns a list of implemented gates\n";
    std::cout << "  quit - stop program\n";
}


void printGates(){
    std::cout << "\nnot \nPauliY \nPauliZ \nPhaseS \nPhaseT \nHadamard \nCNOT - Not implemented \nSWAP - Not implemented \nToffoli - Not implemented"; 
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
        std::cout << "\nCommand: ";  // prompt on a new line
        std::string input;
        std::getline(std::cin, input);

        std::stringstream ss(input);
        std::string command;
        ss >> command;
        command = cleanString(command);

        //std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        if(input == "help"){
            printHelp(); 
        }else if (input == "print"){
            circuit.printCircuitCLI(); 
        }else if (input == "run"){
            executer.execute(circuit, q_state); 
            auto vec = q_state.getStateVector();
            for (const auto& amp : vec) {
                std::cout << "(" << amp.x << ", " << amp.y << ") ";
            }
            std::cout << std::endl;
        }else if (input == "quit"){
            return 0; 
        }else if (command == "gates"){
            printGates(); 
        }else if (command == "add") { 
            std::string gateNameStr;
            int target_qubit = -1 ; 
            int control_qubit = -1; 
            int second_control_qubit = -1;
            
            ss >> gateNameStr; // Read gate name and target qubit
            ss >> target_qubit; 

            if (gateNameStr == "hadamard"){
                circuit.addHadamard(target_qubit); 
            }else if (gateNameStr == "not"){
                std::cout << "added not \n"; 
                circuit.addPualiX(target_qubit); 
            }else if (gateNameStr == "pualiy"){
                circuit.addPauliY(target_qubit); 
            }else if (gateNameStr == "pualiz"){
                circuit.addPauliZ(target_qubit); 
            }else if (gateNameStr == "sphase"){
                circuit.addPhaseS(target_qubit); 
            }else if (gateNameStr == "tphase"){
                circuit.addPhaseT(target_qubit); 
            //}else if (gateNameStr == "cnot"){
            //    ss >> control_qubit; 
            //    circuit.addCNOT(target_qubit, control_qubit); 
            //}else if (gateNameStr == "swap"){
            //    ss >> control_qubit; 
            //    circuit.addSWAP(target_qubit, control_qubit); 
            //}else if (gateNameStr == "toffoli"){
            //    ss >> control_qubit; 
            //    ss >> second_control_qubit; 
            //    circuit.addToffoli(target_qubit, control_qubit, second_control_qubit); 
            }else {
                std::cout<< "invalid command\n";  
            }
            
        }
    }
}
