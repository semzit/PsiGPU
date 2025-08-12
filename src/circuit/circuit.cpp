// In src/circuit/circuit.cpp
#include "../../include/circuit.h"
#include <iostream>
#include "../../include/gates.h"

Circuit::Circuit(int num_qubits)
    :  qubit_count(num_qubits), gates()
{
}

Circuit::~Circuit(){
}

void Circuit::addHadamard(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate hadamard_gate(GateType::Hadamard, target_qubit); 

    gates.push_back(hadamard_gate); 
}

void Circuit::addPualiX(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate puali_x_gate(GateType::PauliX, target_qubit); 

    gates.push_back(puali_x_gate); 
}

void Circuit::addPauliY(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate puali_y_gate(GateType::PauliY, target_qubit); 

    gates.push_back(puali_y_gate); 
}

void Circuit::addPauliZ(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate puali_z_gate(GateType::PauliZ, target_qubit); 

    gates.push_back(puali_z_gate); 
}

void Circuit::addPhaseS(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate phase_s_gate(GateType::PhaseS, target_qubit); 

    gates.push_back(phase_s_gate); 
}

void Circuit::addPhaseT(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate phase_t_gate(GateType::PhaseT, target_qubit); 

    gates.push_back(phase_t_gate); 
}

void Circuit::addRotationX(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate rotation_x(GateType::RotationX, target_qubit); 

    gates.push_back(rotation_x); 
}

void Circuit::addRotationY(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate rotation_y(GateType::RotationY, target_qubit); 

    gates.push_back(rotation_y); 
}

void Circuit::addRotationZ(int target_qubit) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate rotation_z(GateType::RotationZ, target_qubit); 

    gates.push_back(rotation_z); 
}
void Circuit::addCNOT(int target_qubit, int control) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate CNOT_gate(GateType::CNOT, target_qubit, control); 

    gates.push_back(CNOT_gate); 
}
void Circuit::addSWAP(int target_qubit, int control) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate SWAP_gate(GateType::SWAP, target_qubit, control); 

    gates.push_back(SWAP_gate);  
}
void Circuit::addToffoli(int target_qubit, int control, int control_two) {
    if (target_qubit < 0 || target_qubit >= qubit_count){
        throw std::out_of_range("Atempted to add gate to qubit " + std::to_string(target_qubit) +  " but there are only " + std::to_string(qubit_count)); 
    }

    Gate Toffoli_gate(GateType::Toffoli, target_qubit, control, control_two); 

    gates.push_back(Toffoli_gate); 
}


void Circuit::printCircuitCLI() const {
    int qubit_label_width = 2 + std::to_string(qubit_count - 1).length();

  
    std::vector<std::string> circuit_grid(qubit_count, std::string(gates.size() * 4 + 5, '-')); // 4 chars per gate + some initial/final wire
    
    // Adjust initial wire segments
    for (size_t i = 0; i < qubit_count; ++i) {
        circuit_grid[i] = "Q" + std::to_string(i) + ":" + std::string(qubit_label_width - ("Q" + std::to_string(i) + ":").length() + 1, '-') + circuit_grid[i];
        circuit_grid[i] = circuit_grid[i].substr(0, circuit_grid[i].length() - 5) + "-"; // Trim excess and ensure ends with a single dash
    }


    // Populate the grid with gates
    for (size_t col_idx = 0; col_idx < gates.size(); ++col_idx) {
        const Gate& gate = gates[col_idx];
        int gate_start_pos = (col_idx * 4) + qubit_label_width + 2; // Position in string for this gate

        // Reset all lines to '-' for this gate's column
        for(size_t i = 0; i < qubit_count; ++i) {
            if (circuit_grid[i][gate_start_pos] == ' ') circuit_grid[i][gate_start_pos] = '-';
            if (circuit_grid[i][gate_start_pos+1] == ' ') circuit_grid[i][gate_start_pos+1] = '-';
            if (circuit_grid[i][gate_start_pos+2] == ' ') circuit_grid[i][gate_start_pos+2] = '-';
        }

        // Place gate symbols
        switch (gate.type) {
            case GateType::Hadamard:
            case GateType::PauliX:
            case GateType::PauliY:
            case GateType::PauliZ: {
                int q = gate.target_qubit;
                circuit_grid[q][gate_start_pos + 1] = (gate.type == GateType::Hadamard) ? 'H' : 
                                                      ((gate.type == GateType::PauliX) ? 'X' : 
                                                      ((gate.type == GateType::PauliY) ? 'Y' : 'Z'));
                circuit_grid[q][gate_start_pos] = '-';
                circuit_grid[q][gate_start_pos+2] = '-';
                break;
            }
            case GateType::CNOT: {
                int control = gate.control_qubit;
                int target = gate.target_qubit;

                circuit_grid[control][gate_start_pos + 1] = 'C'; // Control
                circuit_grid[target][gate_start_pos + 1] = 'X';  // Target

                // Draw vertical line connecting control and target
                int start_q = std::min(control, target);
                int end_q = std::max(control, target);
                for (int q = start_q + 1; q < end_q; ++q) {
                    circuit_grid[q][gate_start_pos + 1] = '|';
                }
                // Ensure the direct connection points are wires for simplicity, or add specific join symbols
                if (control != target) { // Only draw connector if they're different qubits
                    circuit_grid[start_q][gate_start_pos + 1] = (start_q == control) ? 'C' : 'X';
                    circuit_grid[end_q][gate_start_pos + 1] = (end_q == control) ? 'C' : 'X';
                }
                break;
            }
            case GateType::SWAP: {
                int q1 = gate.control_qubit; // Re-using control_qubit for the first qubit in SWAP
                int q2 = gate.target_qubit;

                circuit_grid[q1][gate_start_pos + 1] = 'X';
                circuit_grid[q2][gate_start_pos + 1] = 'X';

                // Draw vertical line connecting them
                int start_q = std::min(q1, q2);
                int end_q = std::max(q1, q2);
                for (int q = start_q + 1; q < end_q; ++q) {
                    circuit_grid[q][gate_start_pos + 1] = '|';
                }
                break;
            }
            case GateType::Toffoli: {
                int q1 = gate.control_qubit; 
                int q2 = gate.target_qubit;
                int q3 = gate.control_qubit_two; 

                // Draw vertical line connecting them
                int start_q = std::min(q1, q2);
                int end_q = std::max(q1, q2);
                for (int q = start_q + 1; q < end_q; ++q) {
                    circuit_grid[q][gate_start_pos + 1] = '|';
                }

                // Draw vertical line connecting them
                start_q = std::min(q2, q3);
                end_q = std::max(q2, q3);
                for (int q = start_q + 1; q < end_q; ++q) {
                    circuit_grid[q][gate_start_pos + 1] = '|';
                }

                break;
            }
            // Add other multi-qubit gates like Toffoli here
            // case GateType::Toffoli: { /* ... logic for Toffoli ... */ break; }
            default:
                // Handle unsupported gate types or leave as default wire
                break;
        }
    }

    // Print the final grid
    std::cout << "\nCLI Circuit Visualization:\n";
    for (const auto& row_str : circuit_grid) {
        std::cout << row_str << std::endl;
    }
}

