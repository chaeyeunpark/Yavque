#pragma once

#include "../utils.hpp"
#include "Operator.hpp"

namespace yavque
{
namespace detail
{
class HamiltonianImpl
{
private:
	const Eigen::SparseMatrix<cx_double> ham_;

	mutable bool diagonalized_;
	mutable Eigen::VectorXd evals_;
	mutable Eigen::MatrixXcd evecs_;

	void diagonalize() const
	{
		//add mutex
		if(diagonalized_)
		{
			return ;
		}

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Eigen::MatrixXcd{ham_});
		evals_ = es.eigenvalues();
		evecs_ = es.eigenvectors();
		diagonalized_ = true;
	}

public:
	explicit HamiltonianImpl(const Eigen::SparseMatrix<cx_double>& ham)
		: ham_{ham}
	{
		assert(ham_.rows() == ham_.cols()); //check diagonal
		diagonalized_ = false;
	}

	explicit HamiltonianImpl(Eigen::SparseMatrix<cx_double>&& ham)
		: ham_{ham}
	{
		assert(ham_.rows() == ham_.cols()); //check diagonal
		diagonalized_ = false;
	}

	[[nodiscard]] const Eigen::SparseMatrix<cx_double>& get_ham() const&
	{
		return ham_;
	}

	[[nodiscard]] Eigen::SparseMatrix<cx_double> get_ham() &&
	{
		return ham_;
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const 
	{
		return ham_*st;
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
};
} //namespace detail

class Hamiltonian final
	: public Operator
{
private:
	std::shared_ptr<const detail::HamiltonianImpl> p_;
	cx_double constant_ = 1.0;

public:
	explicit Hamiltonian(const Eigen::SparseMatrix<cx_double>& ham,
			const std::string& name = {})
		: Operator(ham.rows(), name), p_{std::make_shared<detail::HamiltonianImpl>(ham)}
	{
		assert(ham.rows() == ham.cols()); //check diagonal
	}

	explicit Hamiltonian(const std::shared_ptr<const detail::HamiltonianImpl>& p,
			const std::string& name = {}, cx_double constant = 1.0) 
		: Operator(p->get_ham().rows(), name), p_{p}, constant_{constant}
	{
	}

	[[nodiscard]] std::shared_ptr<const detail::HamiltonianImpl> get_impl() const
	{
		return p_;
	}

	Hamiltonian(const Hamiltonian& ) = default;
	Hamiltonian(Hamiltonian&& ) = default;

	Hamiltonian& operator=(const Hamiltonian& ) = delete;
	Hamiltonian& operator=(Hamiltonian&& ) = delete;
	~Hamiltonian() override = default;

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto copied = std::make_unique<Hamiltonian>(*this);
		return copied;
	}

	[[nodiscard]] Eigen::SparseMatrix<cx_double> get_ham()  const
	{
		return constant_*p_->get_ham();
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return constant_*p_->apply_right(st);
	}

	[[nodiscard]] const Eigen::MatrixXcd& evecs() const
	{
		return p_->evecs();
	}

	[[nodiscard]] const Eigen::VectorXd& evals() const
	{
		return p_->evals();
	}

	[[nodiscard]] bool is_same_ham(const Hamiltonian& rhs) const
	{
		return (rhs.p_ == p_);
	}

	void dagger_in_place_impl() override
	{
		constant_ = std::conj(constant_);
	}
};

} //namespace yavque
