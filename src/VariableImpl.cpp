#include <memory>

#include "yavque/Circuit.hpp"
#include "yavque/Variable.hpp"

namespace yavque
{

std::string VariableImpl::desc() const
{
	std::ostringstream ss;
	ss << "[" << name_ << "]"
	   << "\n";
	ss << "Derivatives: ";
	for(const auto& circ : deriv_circuits_)
	{
		ss << circ->desc() << "\n";
	}
	return ss.str();
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
	{
		return grad_;
	}

	if(deriv_circuits_.empty())
	{
		return nullptr; // or raise exception
	}

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
	std::vector<std::shared_ptr<Circuit>>{}.swap(deriv_circuits_);
	grad_updated_ = false;
}
} // namespace yavque
