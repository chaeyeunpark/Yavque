#pragma once

#include "../utils.hpp"
#include "Operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <omp.h>

namespace yavque
{

class DenseHermitianMatrix
{
private:
	Eigen::MatrixXcd ham_;

	mutable bool diagonalized_;
	mutable omp_lock_t diagonalize_mutex_;
	mutable Eigen::VectorXd evals_;
	mutable Eigen::MatrixXcd evecs_;

public:
	explicit DenseHermitianMatrix(Eigen::MatrixXcd ham) : ham_{std::move(ham)}
	{
		assert(ham_.rows() == ham_.cols()); // check diagonal
		omp_init_lock(&diagonalize_mutex_);
		diagonalized_ = false;
	}

	~DenseHermitianMatrix() {
		omp_destroy_lock(&diagonalize_mutex_);
	}

	void diagonalize() const
	{
		omp_set_lock(&diagonalize_mutex_);
		if(!diagonalized_)
		{
			omp_unset_lock(&diagonalize_mutex_);
			if(diagonalized_) {
				return;
			}
			const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(ham_);
			evals_ = es.eigenvalues();
			evecs_ = es.eigenvectors();
			diagonalized_ = true;
			omp_unset_lock(&diagonalize_mutex_);
		}
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
