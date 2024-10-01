#pragma once

#include <memory>

#include "DenseHermitianMatrix.hpp"
#include "Operator.hpp"

#include "../utils.hpp"

namespace yavque
{

class SingleQubitOperator final : public Operator
{
private:
	Eigen::MatrixXcd op_;

	// const uint32_t n_qubits_;
	uint32_t qubit_idx_;

	void dagger_in_place_impl() override { op_.adjointInPlace(); }

public:
	explicit SingleQubitOperator(const Eigen::MatrixXcd& op, uint32_t n_qubits,
	                             uint32_t qubit_idx, const std::string& name = {})
		: Operator(1U << n_qubits, name), op_{op}, qubit_idx_{qubit_idx}
	{
		if((op.rows() != 2) || (op.cols() != 2))
		{
			throw std::logic_error("Dimension of the DenseHermitianMatrix must be 2.");
		}
	}

	SingleQubitOperator(const SingleQubitOperator& rhs) = default;
	SingleQubitOperator(SingleQubitOperator&& rhs) = default;

	SingleQubitOperator& operator=(const SingleQubitOperator& rhs) = delete;
	SingleQubitOperator& operator=(SingleQubitOperator&& rhs) = delete;

	~SingleQubitOperator() override = default;

	[[nodiscard]] std::unique_ptr<Operator> clone() const override
	{
		return std::make_unique<SingleQubitOperator>(*this);
	}

	[[nodiscard]] Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		assert(st.size() == dim());

		return apply_single_qubit(st, op_, qubit_idx_);
	}
};
} // namespace yavque
