#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Circuit.hpp"
#include "utils.hpp"

namespace qunn
{
/*
 * @param op must be a hermitian matrix
 * */
std::pair<double, Eigen::VectorXd>
value_and_grad(const Eigen::SparseMatrix<qunn::cx_double>& op, const Circuit& circuit);
};
