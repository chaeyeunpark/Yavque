#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "typedefs.hpp"

namespace yavque
{
Eigen::SparseMatrix<double> pauli_x();
Eigen::SparseMatrix<cx_double> pauli_y();
Eigen::SparseMatrix<double> pauli_z();
Eigen::SparseMatrix<double> pauli_xx();
Eigen::SparseMatrix<double> pauli_yy();
Eigen::SparseMatrix<double> pauli_zz();
} // namespace yavque
