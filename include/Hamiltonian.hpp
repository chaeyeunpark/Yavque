#pragma once

#include "utility.hpp"
#include "Operator.hpp"

namespace qunn
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
			return ;

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Eigen::MatrixXcd{ham_});
		evals_ = es.eigenvalues();
		evecs_ = es.eigenvectors();
		diagonalized_ = true;
	}

public:
	HamiltonianImpl(const Eigen::SparseMatrix<cx_double>& ham)
		: ham_{ham}
	{
		assert(ham_.rows() == ham_.cols()); //check diagonal
		diagonalized_ = false;
	}

	HamiltonianImpl(Eigen::SparseMatrix<cx_double>&& ham)
		: ham_{std::move(ham)}
	{
		assert(ham_.rows() == ham_.cols()); //check diagonal
		diagonalized_ = false;
	}

	const Eigen::SparseMatrix<cx_double>& get_ham() const&
	{
		return ham_;
	}

	Eigen::SparseMatrix<cx_double> get_ham() &&
	{
		return ham_;
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const 
	{
		return ham_*st;
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
};

class Hamiltonian final
	: public Operator
{
private:
	std::shared_ptr<HamiltonianImpl> p_;

public:
	explicit Hamiltonian(const Eigen::SparseMatrix<cx_double>& ham, std::string name_)
		: Operator(ham.rows(), name_), p_{std::make_shared<HamiltonianImpl>(ham)}
	{
		assert(ham.rows() == ham.cols()); //check diagonal
	}

	explicit Hamiltonian(const Eigen::SparseMatrix<cx_double>& ham)
		: Operator(ham.rows()), p_{std::make_shared<HamiltonianImpl>(ham)}
	{
		assert(ham.rows() == ham.cols()); //check diagonal
	}

	explicit Hamiltonian(std::shared_ptr<HamiltonianImpl> p) 
		: Operator(p->get_ham().rows()), p_{p}
	{
	}

	std::shared_ptr<HamiltonianImpl> get_impl() const
	{
		return p_;
	}

	Hamiltonian(const Hamiltonian& ) = default;
	Hamiltonian(Hamiltonian&& ) = default;

	Hamiltonian& operator=(const Hamiltonian& ) = delete;
	Hamiltonian& operator=(Hamiltonian&& ) = delete;

	std::unique_ptr<Operator> clone() const override
	{
		auto copied = std::make_unique<Hamiltonian>(*this);
		copied->set_name(name() + " copied");
		return copied;
	}

	Eigen::SparseMatrix<cx_double> get_ham()  const
	{
		return p_->get_ham();
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return p_->apply_right(st);
	}

	const Eigen::MatrixXcd& evecs() const
	{
		return p_->evecs();
	}

	const Eigen::VectorXd& evals() const
	{
		return p_->evals();
	}

	bool is_same_ham(const Hamiltonian& rhs) const
	{
		return (rhs.p_ == p_);
	}

	void dagger_in_place_impl() override
	{
		//do nothing as Hamiltonian is Hermitian
	}
};

} //namespace qunn
