#pragma once

#include <memory>
#include <exception>
#include "Operators/Operator.hpp"
#include "Utilities/apply_operator.hpp"

#include "Operators/DenseHermitianMatrix.hpp"

namespace qunn
{

class TwoQubitOperator
	: public Operator
{
private:
	Eigen::MatrixXcd op_;

	const uint32_t n_qubits_;
	const uint32_t i_;
	const uint32_t j_;

	void dagger_in_place_impl() override
	{
		op_.adjointInPlace();
	}

public:
	explicit TwoQubitOperator(const Eigen::MatrixXcd& op, 
			uint32_t n_qubits, uint32_t i, uint32_t j,std::string name = {})
		: Operator(1u << n_qubits, std::move(name)), op_{op}, n_qubits_{n_qubits}, 
		i_{i}, j_{j}
	{
		if((op.rows() != 4) || (op.cols() != 4))
		{
			throw std::logic_error("Dimension of the DenseHermitianMatrix must be 4.");
		}
	}

	TwoQubitOperator(const TwoQubitOperator& rhs) = default;

	std::unique_ptr<Operator> clone() const override
	{
		return std::make_unique<TwoQubitOperator>(*this);
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		assert(st.size() == dim());

		return apply_two_qubit(st, op_, {i_, j_});
	}
};
}
