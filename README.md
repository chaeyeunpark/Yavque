# Yavque
Yet another variational quantum (eigensolver) library. 

# Objective
**Yavque** aims to provide an efficient classical simulation of various VQE circuits. In addition, it offers an autograd-like interface. 

Many Hamiltonians acting on a qubit array are sum of commuting few body terms. For example, consider the Hamiltonian <img src="https://latex.codecogs.com/gif.latex?H%3D%5Csum_%7B%5Clangle%20i%2Cj%20%5Crangle%7D%20h_%7Bi%2Cj%7D%3D%5Csum_%7B%5Clangle%20i%2Cj%20%5Crangle%7D%20J_%7Bij%7D%20Z_i%20Z_j">. All terms are diagonal thus the time evolution can be easily applied as <img src="https://latex.codecogs.com/gif.latex?%5Clangle%20x%7C%20e%5E%7B-iHt%7D%20%7C%20%5Cpsi%20%5Crangle%20%3D%20e%5E%7B-i%20H%28x%29%20t%7D%20%5Cpsi%28x%29">. 

Even when operators are not diagonal, time evolution operator can be much efficiently done by each evolution sequentially. Three classes (`qunn::DiagonalOperator`, `qunn::SumLocalHam`, and `qunn::SumPauliString`) in Yavque implement such Hamiltonians with correcspongding time evolution operators (`qunn::DiagonalHamEvol`, `qunn::ProductHamEvol`, and `qunn::SumPauliStringHamEvol`). 

# Documents?
Hope to be documented soon...
