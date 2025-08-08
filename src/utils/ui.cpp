#include "ui.h"
#include <iostream>
#include <cctype>
#include <string>
#include <algorithm>

void printHelp(){
    std::cout << "Commands: \n addQubit \n addQubits \n addCNOT \n addSWAP \n"; 
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