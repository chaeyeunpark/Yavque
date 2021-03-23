#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace qunn
{
Eigen::SparseMatrix<double> pauli_x();
Eigen::SparseMatrix<double> pauli_z();
Eigen::SparseMatrix<double> pauli_xx();
Eigen::SparseMatrix<double> pauli_yy();
Eigen::SparseMatrix<double> pauli_zz();
}// namespace qunn
