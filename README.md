# Yavque
Yet another variational quantum (eigensolver) library. 

## Objective
**Yavque** aims to provide an efficient classical simulation of various variational quantum circuits with an autograd-like interface. 

Lots of variational circuit utilizes the time evolution of some Hamiltonians. In the exterem case, when the Hamiltonian is diagonal in the computational basis, the corresponding time evolution is also diagonal. Even when the Hamiltonian is not diagonal, time evolution operator can be much efficiently done when it consists of few-body terms and it .
Classes (e.g. `yavque::DiagonalOperator`, `yavque::SumLocalHam`, and `yavque::SumPauliString`) in Yavque implement such Hamiltonians with correcspongding time evolution operators (`yavque::DiagonalHamEvol`, `yavque::SumLocalHamEvol`, and `qunn::SumPauliStringHamEvol`). 

## Documents?
Hope to be documented soon...
