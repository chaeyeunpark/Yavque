#pragma once
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "operators.hpp"

namespace yavque
{
class Variable;

class Circuit
{
private:
	uint32_t dim_;

	// Operators
	std::vector<std::unique_ptr<Operator>> ops_;

	// States evaluated from the left
	mutable std::vector<std::shared_ptr<const Eigen::VectorXcd>> states_from_left_;
	mutable uint32_t states_updated_to_ = 0;

	void add_op_right(std::unique_ptr<Operator> op) { ops_.emplace_back(std::move(op)); }

	void add_op_left(std::unique_ptr<Operator> op)
	{
		states_updated_to_ = 0;
		ops_.emplace(ops_.begin(), std::move(op));
	}

protected:
	template<typename ConstIterator>
	Circuit(uint32_t dim, const ConstIterator& begin, const ConstIterator& end)
		: dim_{dim}
	{
		// check all ops has the same dim
		for(auto iter = begin; iter != end; ++iter)
		{
			ops_.push_back((*iter)->clone());
		}
	}

public:
	explicit Circuit(uint32_t dim) : dim_{dim} { }

	Circuit(const Circuit& rhs); // Defined in cpp
	Circuit(Circuit&& rhs) = default;

	Circuit& operator=(const Circuit& rhs); // Defined in cpp
	Circuit& operator=(Circuit&& rhs) = default;

	~Circuit() = default;

	uint32_t get_dim() const { return dim_; }

	bool is_evaluated() const { return (states_updated_to_ == ops_.size() + 1); }

	// evaluate
	void evaluate() const
	{
		// Initial state hasn't been set
		if(states_updated_to_ == 0)
		{
			return;
		}

		// All states are evaluated
		if(states_updated_to_ == ops_.size() + 1)
		{
			return;
		}

		for(auto d = states_from_left_.size() - 1; d < ops_.size(); ++d)
		{
			states_from_left_.emplace_back(std::make_shared<Eigen::VectorXcd>(
				ops_[d]->apply_right(*states_from_left_[d])));
		}
		states_updated_to_ = ops_.size() + 1;
	}

	void clear_evaluated()
	{
		states_updated_to_ = 1;
		states_from_left_.resize(1);
	}

	std::size_t num_operators() const { return ops_.size(); }

	const std::unique_ptr<Operator>& operator_at(std::size_t idx) const
	{
		return ops_[idx];
	}

	template<typename T, typename... Args> void add_op_right(Args&&... args)
	{
		static_assert(std::is_convertible_v<T*, Operator*>,
		              "T must be a subclass of yavque::Operator.");
		ops_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
	}

	template<typename T, typename... Args> void add_op_left(Args&&... args)
	{
		static_assert(std::is_convertible_v<T*, Operator*>,
		              "T must be a subclass of yavque::Operator.");
		states_updated_to_ = 0;
		ops_.emplace(ops_.begin(), std::make_unique<T>(std::forward<Args>(args)...));
	}

	/**
	 * evaluate |st> - C
	 * When C = U_N \cdots U_1, it computes U_N \cdots U_1 | st \rangle
	 * */
	void set_input(const Eigen::VectorXcd& st)
	{
		states_updated_to_ = 1;
		states_from_left_.clear();
		states_from_left_.push_back(std::make_shared<Eigen::VectorXcd>(st));
	}

	/**
	 * subcircuit from 0 to d [0,d)
	 */
	Circuit to(uint32_t d) const
	{
		Circuit circ(dim_, ops_.begin(), ops_.begin() + d);
		if(states_updated_to_ != 0)
		{
			circ.states_from_left_ = std::vector<std::shared_ptr<const Eigen::VectorXcd>>(
				states_from_left_.begin(),
				states_from_left_.begin() + std::min(d + 1, states_updated_to_));
			circ.states_updated_to_ = std::min(d + 1, states_updated_to_);
		}
		return circ;
	}

	/**
	 * subcircuit from d to end [d,end)
	 */
	Circuit from(uint32_t d) const
	{
		Circuit circ(dim_, ops_.begin() + d, ops_.end());
		if(states_updated_to_ > d)
		{
			circ.states_from_left_ = std::vector<std::shared_ptr<const Eigen::VectorXcd>>(
				states_from_left_.begin() + d,
				states_from_left_.begin() + states_updated_to_);
			circ.states_updated_to_ = states_updated_to_ - d;
		}

		return circ;
	}

	Circuit& operator|=(const Circuit& b)
	{
		using std::begin;
		using std::end;
		assert(dim_ == b.dim_);
		for(const auto& op : b.ops_)
		{
			ops_.push_back(op->clone());
		}
		return *this;
	}

	// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
	Circuit& operator|=(Circuit&& b)
	{
		using std::begin;
		using std::end;
		assert(dim_ == b.dim_);
		for(auto&& op : b.ops_)
		{
			ops_.emplace_back(std::move(op));
		}
		b.states_updated_to_ = 0;
		b.states_from_left_.resize(0);
		b.ops_.resize(0);
		return *this;
	}

	Circuit dagger() const
	{
		Circuit res(dim_, ops_.crbegin(), ops_.crend());
		for(std::size_t idx = 0; idx < ops_.size(); ++idx)
		{
			res.ops_[idx]->dagger_in_place();
		}
		return res;
	}

	std::vector<Variable> variables() const;

	void derivs() const;

	std::shared_ptr<const Eigen::VectorXcd> state_at(uint32_t idx) const
	{
		// May throw exception instead
		if(idx > ops_.size())
		{
			return nullptr;
		}

		// If not all stastes are evaluated
		if(idx > states_updated_to_)
		{
			evaluate();
		}

		return states_from_left_[idx];
	}

	std::shared_ptr<const Eigen::VectorXcd> output() const
	{
		if(!is_evaluated())
		{
			evaluate();
		}
		return states_from_left_.back();
	}

	[[nodiscard]] std::string desc() const
	{
		// change to format in C++20
		std::ostringstream ss;
		for(uint32_t d = 0; d < ops_.size(); ++d)
		{
			ss << "layer" << d << ": [" << ops_[d]->desc() << "]\n";
		}
		return ss.str();
	}
};

Circuit operator|(Circuit a, const Circuit& b);
Circuit operator|(Circuit a, Circuit&& b);

} // namespace yavque
