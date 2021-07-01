# Yavque
Yet another variational quantum (eigensolver) library. 

## Objective
**Yavque** aims to provide an efficient classical simulation of various variational quantum circuits with an autograd-like interface. 


Lots of variational circuits utilize the time evolution of some Hamiltonians. In the extreme case, when the Hamiltonian is diagonal in the computational basis, the corresponding time evolution is also diagonal. One can deal with such time evolution operators much efficiently. Even when the Hamiltonian is not diagonal, the time evolution operator can be much efficiently done when it consists of few-body terms. Classes (e.g. `yavque::DiagonalOperator`, `yavque::SumLocalHam`, and `yavque::SumPauliString`) in Yavque implement such Hamiltonians with corresponding time evolution operators (`yavque::DiagonalHamEvol`, `yavque::SumLocalHamEvol`, and `yavque::SumPauliStringHamEvol`).

## Compile
Currently, the library depends on the Intel MKL and TBB that can be installed from here. You can build the library and examples as:

```bash
$ git clone --recursive https://github.com/chaeyeunpark/Yavque.git
$ cd Yavque
$ mkdir build && cd build
$ cmake -DBUILD_EXAMPLES=ON ..
$ make all
```

In the examples, you can find three different implementations of variational quantum eigensolvers (VQE) with the quantum alternating operator Ansatz for the transverse field Ising model: `qaoa_tfi_diag`, `qaoa_tfi_pauli`, and `qaoa_tfi_ti`, and the hardware efficient Ansatz for solving a chemical Hamiltonian `solve_chem`. As we included sample Hamiltonians for H<sub>2</sub> and NH<sub>3</sub> molecules, you can test run them.


```bash
$ ./solve_chem ../examples/H2.dat 5     # 5 is the circuit depth we want to use
$ ./solve_chem ../examples/NH3.dat 10   # It takes some initial time to load the Hamiltonian 
```
