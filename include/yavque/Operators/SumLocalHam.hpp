#pragma once

#include <edlib/EDP/ConstructSparseMat.hpp>
#include <edlib/EDP/LocalHamiltonian.hpp>

#include "Operator.hpp"

#include "../utils.hpp"

namespace yavque
{
namespace detail
{
	class SumLocalHamImpl
	{
	private:
		uint32_t num_qubits_;
		Eigen::SparseMatrix<cx_double> local_ham_;
		Eigen::SparseMatrix<cx_double> full_ham_;

		mutable bool diagonalized_ = false;
		mutable Eigen::MatrixXcd evecs_;
		mutable Eigen::VectorXd evals_;

		void construct_full_ham()
		{
			edp::LocalHamiltonian<cx_double> ham_ct(num_qubits_, 2);
			for(uint32_t k = 0; k < num_qubits_; ++k)
			{
				ham_ct.addOneSiteTerm(k, local_ham_);
			}
			full_ham_ = edp::constructSparseMat<cx_double>(1U << num_qubits_, ham_ct);
		}

	public:
		SumLocalHamImpl(uint32_t num_qubits,
		                const Eigen::SparseMatrix<cx_double>& local_ham)
			: num_qubits_{num_qubits}, local_ham_{local_ham}
		{
			assert(local_ham.cols() == 2);
			assert(local_ham.rows() == 2);
			construct_full_ham();
		}

		[[nodiscard]] const Eigen::SparseMatrix<cx_double>& get_local_ham() const&
		{
			return local_ham_;
		}

		[[nodiscard]] Eigen::SparseMatrix<cx_double> get_local_ham() &&
		{
			return local_ham_;
		}

		[[nodiscard]] const Eigen::SparseMatrix<cx_double>& get_full_ham() const&
		{
			return full_ham_;
		}

		[[nodiscard]] Eigen::SparseMatrix<cx_double> get_full_ham() &&
		{
			return full_ham_;
		}

		[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const
		{
			return full_ham_ * st;
		}

		[[nodiscard]] uint32_t dim() const { return (1U << num_qubits_); }

		[[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }

		/*
		 * compute exp(x*local_ham_) */
		[[nodiscard]] Eigen::MatrixXcd local_ham_exp(cx_double x) const
		{
			if(!diagonalized_)
			{
				Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(local_ham_);
				evecs_ = es.eigenvectors();
				evals_ = es.eigenvalues();
				diagonalized_ = true;
			}
			Eigen::VectorXcd v = exp(x * evals_.array());
			return evecs_ * v.asDiagonal() * evecs_.adjoint();
		}
	};
} // namespace detail

class SumLocalHam final : public Operator
{
private:
	std::shared_ptr<const detail::SumLocalHamImpl> p_;
	cx_double constant_ = 1.0;

public:
	explicit SumLocalHam(uint32_t num_qubits, const Eigen::SparseMatrix<cx_double>& ham,
	                     const std::string& name = {})
		: Operator(ham.rows(), name), p_{std::make_shared<const detail::SumLocalHamImpl>(
										  num_qubits, ham)}
	{
		assert(ham.rows() == ham.cols()); // check diagonal
	}

	explicit SumLocalHam(std::shared_ptr<const detail::SumLocalHamImpl> p,
	                     const std::string& name = {}, cx_double constant = 1.0)
		: Operator(p->dim(), name), p_{std::move(p)}, constant_{constant}
	{
	}

	[[nodiscard]] std::shared_ptr<const detail::SumLocalHamImpl> get_impl() const
	{
		return p_;
	}

	SumLocalHam(const SumLocalHam&) = default;
	SumLocalHam(SumLocalHam&&) = default;

	SumLocalHam& operator=(const SumLocalHam&) = delete;
	SumLocalHam& operator=(SumLocalHam&&) = delete;

	~SumLocalHam() override = default;

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto copied = std::make_unique<SumLocalHam>(*this);
		return copied;
	}

	[[nodiscard]] Eigen::SparseMatrix<cx_double> get_ham() const
	{
		return constant_ * p_->get_full_ham();
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return constant_ * p_->apply_right(st);
	}

	[[nodiscard]] bool is_same_ham(const SumLocalHam& rhs) const
	{
		return (rhs.p_ == p_);
	}

	void dagger_in_place_impl() override { constant_ = std::conj(constant_); }
};
} // namespace yavque
