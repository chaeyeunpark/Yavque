#pragma once
#include <atomic>
#include <memory>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include "Counter.hpp"
#include "VariableImpl.hpp"

namespace yavque
{
class Circuit;


class Variable
{
private:
	std::shared_ptr<VariableImpl> p_ {std::make_shared<VariableImpl>(0.0)};

public:
	Variable(const Variable& rhs) = default;
	Variable(Variable&& rhs) = default;

	Variable& operator=(const Variable& ) = delete;
	Variable& operator=(Variable&& rhs) noexcept
	{
		p_ = std::move(rhs.p_);
		return *this;
	}

	explicit Variable(double value)
		: p_{std::make_shared<VariableImpl>(value)}
	{
	}

	explicit Variable(double value, std::string name)
		: p_{std::make_shared<VariableImpl>(value, std::move(name))}
	{
	}

	void change_to(const Variable& rhs)
	{
		p_ = rhs.p_;
	}

	bool operator==(const Variable& rhs) const
	{
		return p_ == rhs.p_;
	}
	bool operator!=(const Variable& rhs) const
	{
		return p_ != rhs.p_;
	}


	[[nodiscard]] std::string name() const
	{
		return p_->name();
	}

	[[nodiscard]] std::string desc() const
	{
		return p_->desc();
	}

	[[nodiscard]] double value() const
	{
		return p_->value();
	}

	void set_value(double new_value)
	{
		p_->set_value(new_value);
	}

	Variable& operator=(double v)
	{
		p_->set_value(v);
		return *this;
	}

	Variable& operator+=(double v) 
	{
		p_->set_value(value() + v);
		return *this;
	}

	Variable& operator-=(double v) 
	{
		p_->set_value(value() - v);
		return *this;
	}

	Variable& operator*=(double v) 
	{
		p_->set_value(value() * v);
		return *this;
	}

	friend Variable operator+(double lhs, const Variable& rhs) 
	{
		return Variable(lhs + rhs.value());
	};

	friend Variable operator+(const Variable& lhs, double rhs) 
	{
		return Variable(lhs.value() + rhs);
	};

	friend Variable operator-(double lhs, const Variable& rhs) 
	{
		return Variable(lhs - rhs.value());
	};

	friend Variable operator-(const Variable& lhs, double rhs) 
	{
		return Variable(lhs.value() - rhs);
	};


	friend Variable operator*(double lhs, const Variable& rhs) 
	{
		return Variable(lhs * rhs.value());
	};

	friend Variable operator*(const Variable& lhs, double rhs) 
	{
		return Variable(lhs.value() * rhs);
	};


	friend Variable operator/(double lhs, const Variable& rhs) 
	{
		return Variable(lhs / rhs.value());
	};

	friend Variable operator/(const Variable& lhs, double rhs) 
	{
		return Variable(lhs.value() / rhs);
	};

	void add_grad(const Circuit& circ)
	{
		p_->add_grad(circ);
	}

	void add_grad(Circuit&& circ)
	{
		p_->add_grad(std::move(circ));
	}

	[[nodiscard]] std::shared_ptr<const Eigen::VectorXcd> grad() const
	{
		return p_->grad();
	}

	void zero_grad()
	{
		p_->zero_grad();
	}

	~Variable() = default;
};

} // namespace yavque
