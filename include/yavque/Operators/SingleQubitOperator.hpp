#pragma once

#include <memory>
#include "Operator.hpp"
#include "../utils.hpp"

#include "DenseHermitianMatrix.hpp"

namespace yavque
{

class SingleQubitOperator final
	: public Operator
{
private:
	Eigen::MatrixXcd op_;

	const uint32_t n_qubits_;
	const uint32_t qubit_idx_;

	void dagger_in_place_impl() override
	{
		op_.adjointInPlace();
	}

public:
	explicit SingleQubitOperator(const Eigen::MatrixXcd& op, 
			uint32_t n_qubits, uint32_t qubit_idx, std::string name = {})
		: Operator(1u << n_qubits, std::move(name)), op_{op}, n_qubits_{n_qubits}, 
		qubit_idx_{qubit_idx}
	{	
		if((op.rows() != 2) || (op.cols() != 2))
		{
			throw std::logic_error("Dimension of the DenseHermitianMatrix must be 2.");
		}
	}

	SingleQubitOperator(const SingleQubitOperator& rhs) = default;

	std::unique_ptr<Operator> clone() const override
	{
		return std::make_unique<SingleQubitOperator>(*this);
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		assert(st.size() == dim());

		return apply_single_qubit(st, op_, qubit_idx_);
	}
};
}
