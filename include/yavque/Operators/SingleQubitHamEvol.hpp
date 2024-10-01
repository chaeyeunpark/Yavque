#pragma once

#include <exception>
#include <memory>
#include <sstream>

#include "../Univariate.hpp"
#include "../Variable.hpp"
#include "../utils.hpp"

#include "DenseHermitianMatrix.hpp"
#include "Hamiltonian.hpp"
#include "Operator.hpp"
#include "SingleQubitOperator.hpp"

namespace yavque
{

class SingleQubitHamEvol final : public Operator, public Univariate
{
private:
	std::shared_ptr<const DenseHermitianMatrix> ham_;
	uint32_t n_qubits_;
	uint32_t qubit_idx_;
	bool conjugate_ = false;

	void dagger_in_place_impl() override { conjugate_ = !conjugate_; }

public:
	explicit SingleQubitHamEvol(const std::shared_ptr<const DenseHermitianMatrix>& ham,
	                            uint32_t n_qubits, uint32_t qubit_idx,
	                            const std::string& name = {})
		: Operator(1U << n_qubits, name), ham_{ham}, n_qubits_{n_qubits},
		  qubit_idx_{qubit_idx}
	{
		if(ham->dim() != 2)
		{
			throw std::logic_error("Dimension of the DenseHermitianMatrix must be 2.");
		}
	}

	SingleQubitHamEvol(SingleQubitHamEvol&&) = default;
	SingleQubitHamEvol(const SingleQubitHamEvol&) = default;
	SingleQubitHamEvol& operator=(const SingleQubitHamEvol&) = delete;
	SingleQubitHamEvol& operator=(SingleQubitHamEvol&&) = delete;

	~SingleQubitHamEvol() override = default;

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		auto p = std::make_unique<SingleQubitHamEvol>(*this);
		p->change_variable(Variable{var_.value()});
		return p;
	}

	[[nodiscard]] std::unique_ptr<Operator> log_deriv() const override
	{
		constexpr std::complex<double> I(0., 1.0);
		const cx_double constant = conjugate_ ? I : -I;

		std::ostringstream os;
		os << "derivative of (" << name() << ")";
		return std::make_unique<SingleQubitOperator>(constant * ham_->get_ham(),
		                                             n_qubits_, qubit_idx_, os.str());
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		constexpr std::complex<double> I(0., 1.0);
		assert(dim() == st.size());
		cx_double x = -I * var_.value();
		if(conjugate_)
		{
			x = -x;
		}
		const Eigen::MatrixXcd expm = ham_->ham_exp(x);

		return apply_single_qubit(st, expm, qubit_idx_);
	}
};
} // namespace yavque
