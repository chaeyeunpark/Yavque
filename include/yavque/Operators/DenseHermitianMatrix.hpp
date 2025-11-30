#pragma once

#include "../utils.hpp"
#include "Operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <tbb/mutex.h>

namespace yavque
{

class DenseHermitianMatrix
{
private:
	Eigen::MatrixXcd ham_;

	mutable bool diagonalized_;
	mutable tbb::mutex diagonalize_mutex_;
	mutable Eigen::VectorXd evals_;
	mutable Eigen::MatrixXcd evecs_;

	void diagonalize() const
	{
		if(!diagonalized_)
		{
			diagonalize_mutex_.lock();
			if(diagonalized_) {
				diagonalize_mutex_.unlock();
				return;
			}
			const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(ham_);
			evals_ = es.eigenvalues();
			evecs_ = es.eigenvectors();
			diagonalized_ = true;
			diagonalize_mutex_.unlock();
		}
	}

public:
	explicit DenseHermitianMatrix(Eigen::MatrixXcd ham) : ham_{std::move(ham)}
	{
		assert(ham_.rows() == ham_.cols()); // check diagonal
		diagonalized_ = false;
	}

	[[nodiscard]] uint32_t dim() const { return ham_.rows(); }

	[[nodiscard]] const Eigen::MatrixXcd& get_ham() const& { return ham_; }

	[[nodiscard]] Eigen::MatrixXcd get_ham() && { return ham_; }

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

		const Eigen::VectorXcd v = exp(x * evals_.array());
		return evecs_ * v.asDiagonal() * evecs_.adjoint();
	}
};

} // namespace yavque
