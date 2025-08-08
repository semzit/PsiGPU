#include <iostream>
#include <string>
#include "utils/ui.cpp"

int main(){
    std::cout << "Welcome to PsiGPU!" << "\n"; 
    while (true){
        std::string input ; 
        std::cin >> input ; 
        input = cleanString(input);
        
        if(input == "help"){
            printHelp(); 
        }
    }
     
}