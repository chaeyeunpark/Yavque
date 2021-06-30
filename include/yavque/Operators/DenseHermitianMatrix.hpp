#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "../utils.hpp"
#include "Operator.hpp"

namespace yavque
{

class DenseHermitianMatrix
{
private:
	const Eigen::MatrixXcd ham_;

	mutable bool diagonalized_;
	mutable Eigen::VectorXd evals_;
	mutable Eigen::MatrixXcd evecs_;

	void diagonalize() const
	{
		//add mutex
		if(diagonalized_)
			return ;

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(ham_);
		evals_ = es.eigenvalues();
		evecs_ = es.eigenvectors();
		diagonalized_ = true;
	}

public:
	DenseHermitianMatrix(const Eigen::MatrixXcd& ham)
		: ham_{ham}
	{
		assert(ham_.rows() == ham_.cols()); //check diagonal
		diagonalized_ = false;
	}

	uint32_t dim() const
	{
		return ham_.rows();
	}

	const Eigen::MatrixXcd& get_ham() const&
	{
		return ham_;
	}

	Eigen::MatrixXcd get_ham() &&
	{
		return ham_;
	}

	const Eigen::MatrixXcd& evecs() const&
	{
		if(!diagonalized_)
			diagonalize();
		return evecs_;
	}

	Eigen::MatrixXcd evecs() &&
	{
		if(!diagonalized_)
			diagonalize();
		return evecs_;
	}

	const Eigen::VectorXd& evals() const&
	{
		if(!diagonalized_)
			diagonalize();
		return evals_;
	}

	Eigen::VectorXd evals() &&
	{
		if(!diagonalized_)
			diagonalize();
		return evals_;
	}

	Eigen::MatrixXcd ham_exp(cx_double x) const
	{
		if(!diagonalized_)
		{
			diagonalize();
		}
		
		Eigen::VectorXcd v = exp(x*evals_.array());
		return evecs_*v.asDiagonal()*evecs_.adjoint();
	}
};

} //namespace yavque
