#include "yavque/Utilities/pauli_operators.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace yavque
{
Eigen::SparseMatrix<double> pauli_x()
{
	std::vector<Eigen::Triplet<double>> t{{1, 0, 1.0}, {0, 1, 1.0}};
	Eigen::SparseMatrix<double> res(2, 2);
	res.setFromTriplets(t.begin(), t.end());
	return res;
}
Eigen::SparseMatrix<cx_double> pauli_y()
{
	constexpr cx_double I(0., 1.);
	std::vector<Eigen::Triplet<cx_double>> t{{1, 0, I}, {0, 1, -I}};
	Eigen::SparseMatrix<cx_double> res(2, 2);
	res.setFromTriplets(t.begin(), t.end());
	return res;
}

Eigen::SparseMatrix<double> pauli_z()
{
	std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {1, 1, -1.0}};
	Eigen::SparseMatrix<double> res(2, 2);
	res.setFromTriplets(t.begin(), t.end());
	return res;
}

Eigen::SparseMatrix<double> pauli_xx()
{
	Eigen::SparseMatrix<double> res(4, 4);
	res.coeffRef(0, 3) = 1.0;
	res.coeffRef(1, 2) = 1.0;
	res.coeffRef(2, 1) = 1.0;
	res.coeffRef(3, 0) = 1.0;

	res.makeCompressed();
	return res;
}

Eigen::SparseMatrix<double> pauli_yy()
{
	Eigen::SparseMatrix<double> res(4, 4);
	res.coeffRef(0, 3) = -1.0;
	res.coeffRef(1, 2) = 1.0;
	res.coeffRef(2, 1) = 1.0;
	res.coeffRef(3, 0) = -1.0;

	res.makeCompressed();
	return res;
}

Eigen::SparseMatrix<double> pauli_xx_yy()
{
	std::vector<Eigen::Triplet<double>> t{{2, 1, 2.0}, {1, 2, 2.0}};
	Eigen::SparseMatrix<double> res(4, 4);
	res.setFromTriplets(t.begin(), t.end());
	return res;
}

Eigen::SparseMatrix<double> pauli_zz()
{
	Eigen::SparseMatrix<double> res(4, 4);
	res.coeffRef(0, 0) = 1.0;
	res.coeffRef(1, 1) = -1.0;
	res.coeffRef(2, 2) = -1.0;
	res.coeffRef(3, 3) = 1.0;

	res.makeCompressed();
	return res;
}

Eigen::SparseMatrix<double> pauli_xx_yy_zz()
{
	std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {1,1,-1.0}, {2, 1, 2.0}, {1, 2, 2.0}, {2,2, -1.0},{3,3,1.0}};
	Eigen::SparseMatrix<double> res(4, 4);
	res.setFromTriplets(t.begin(), t.end());
	return res;
}

} // namespace yavque
