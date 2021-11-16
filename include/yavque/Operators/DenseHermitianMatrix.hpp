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
		// Add mutex (future)
		if(diagonalized_)
		{
			return ;
		}

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(ham_);
		evals_ = es.eigenvalues();
		evecs_ = es.eigenvectors();
		diagonalized_ = true;
	}

public:
	explicit DenseHermitianMatrix(Eigen::MatrixXcd ham)
		: ham_{std::move(ham)}
	{
		assert(ham_.rows() == ham_.cols()); //check diagonal
		diagonalized_ = false;
	}

	[[nodiscard]] uint32_t dim() const
	{
		return ham_.rows();
	}

	[[nodiscard]] const Eigen::MatrixXcd& get_ham() const&
	{
		return ham_;
	}

	[[nodiscard]] Eigen::MatrixXcd get_ham() &&
	{
		return ham_;
	}

	[[nodiscard]] const Eigen::MatrixXcd& evecs() const&
	{
		if(!diagonalized_)
		{
			diagonalize();
		}
		return evecs_;
	}

	[[nodiscard]] Eigen::MatrixXcd evecs() &&
	{
		if(!diagonalized_)
		{
			diagonalize();
		}
		return evecs_;
	}

	[[nodiscard]] const Eigen::VectorXd& evals() const&
	{
		if(!diagonalized_)
		{
			diagonalize();
		}
		return evals_;
	}

	[[nodiscard]] Eigen::VectorXd evals() &&
	{
		if(!diagonalized_)
		{
			diagonalize();
		}
		return evals_;
	}

	[[nodiscard]] Eigen::MatrixXcd ham_exp(cx_double x) const
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
