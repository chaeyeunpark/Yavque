#include <memory>
#include <sstream>

#include "yavque/Variable.hpp"
#include "yavque/Circuit.hpp"

namespace yavque
{
VariableImpl::VariableImpl(double value)
	: value_{value}
{
	static Counter counter;
	std::ostringstream ss; //change to format C++20
	ss << "variable " << counter.count();
	name_ = ss.str();
}

std::string VariableImpl::desc() const
{
	std::ostringstream ss;
	ss << "[" << name_ << "]" << "\n";
	ss << "Derivatives: " ;
	for(auto circ: deriv_circuits_)
	{
		ss << circ->desc() << "\n";
	}
	return ss.str();
}
	
VariableImpl::VariableImpl(double value, std::string name)
	: value_{value}, name_{std::move(name)}
{
}

double VariableImpl::value() const
{
	return value_;
}

void VariableImpl::set_value(double new_value) 
{
	value_ = new_value;
}

std::string VariableImpl::name() const
{
	return name_;
}

void VariableImpl::set_name(std::string name)
{
	name_ = std::move(name);
}

void VariableImpl::add_grad(const Circuit& circ)
{
	deriv_circuits_.emplace_back(std::make_shared<Circuit>(circ));
	grad_updated_ = false;
}

void VariableImpl::add_grad(Circuit&& circ)
{
	deriv_circuits_.emplace_back(std::make_shared<Circuit>(std::move(circ)));
	grad_updated_ = false;
}

std::shared_ptr<const Eigen::VectorXcd> VariableImpl::grad() const
{
	if(grad_updated_)
		return grad_;

	if(deriv_circuits_.empty())
		return nullptr; //or raise exception

	grad_ = std::make_shared<Eigen::VectorXcd>(*(deriv_circuits_[0]->output()));

	for(std::size_t n = 1; n < deriv_circuits_.size(); ++n)
	{
		*grad_ += *deriv_circuits_[n]->output();
	}
	grad_updated_ = true;

	return std::const_pointer_cast<Eigen::VectorXcd>(grad_);
}

void VariableImpl::zero_grad()
{
	std::vector<std::shared_ptr<Circuit> >{}.swap(deriv_circuits_);
	grad_updated_ = false;
}
}// namespace yavque
