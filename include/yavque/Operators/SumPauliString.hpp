#pragma once

#include <set>
#include <Eigen/Eigenvalues>

#include "../utils.hpp"

#include "Operator.hpp"
#include "CompressedPauliString.hpp"

namespace yavque
{

namespace detail
{

bool commute(const std::map<uint32_t, Pauli>& p1, const std::map<uint32_t, Pauli>& p2);
std::string extract_pauli_string(const std::map<uint32_t, Pauli>& pmap);

class SumPauliStringImpl
{
public:
	using PauliString = std::map<uint32_t, Pauli>;

private:
	const uint32_t num_qubits_;
	std::vector<PauliString> pauli_strings_;
	mutable std::vector<std::shared_ptr<detail::CompressedPauliString> > cps_;

	void update_cps() const
	{
		for(std::size_t n = cps_.size(); n < pauli_strings_.size(); ++n)
		{
			cps_.emplace_back(CPSFactory::get_instance().get_pauli_string_for(
					extract_pauli_string(pauli_strings_[n])));
		}
	}

public:
	SumPauliStringImpl(uint32_t num_qubits)
		: num_qubits_{num_qubits}
	{
	}
	
	explicit SumPauliStringImpl(uint32_t num_qubits, 
			const std::vector<PauliString>& pauli_strings)
		: num_qubits_{num_qubits}, pauli_strings_{pauli_strings}
	{
		//check sites pauli strings applied < num_qubits
	}

	explicit SumPauliStringImpl(uint32_t num_qubits, 
			std::vector<PauliString>&& pauli_strings)
		: num_qubits_{num_qubits}, pauli_strings_{std::move(pauli_strings)}
	{
	}

	uint32_t num_qubits() const
	{
		return num_qubits_;
	}

	void add(const std::map<uint32_t, Pauli>& rhs)
	{
		//check sites pauli strings applied < num_qubits
		pauli_strings_.push_back(rhs);
	}


	void add(std::map<uint32_t, Pauli>&& rhs)
	{
		//check sites pauli strings applied < num_qubits
		pauli_strings_.emplace_back(std::move(rhs));
	}

	bool mutually_commuting() const
	{
		for(uint32_t i = 0; i < pauli_strings_.size()-1; ++i)
		{
			for(uint32_t j = i+1; j < pauli_strings_.size(); ++j)
			{
				if(! commute(pauli_strings_[i], pauli_strings_[j]))
					return false;
			}
		}
		return true;
	}

	Eigen::VectorXcd apply(const Eigen::VectorXcd& vec) const
	{
		assert(vec.size() == (1u << num_qubits_));

		if(cps_.size() < pauli_strings_.size())
			update_cps();

		Eigen::VectorXcd res = Eigen::VectorXcd::Zero(vec.size());
		for(std::size_t n = 0; n < pauli_strings_.size(); ++n)
		{
			const auto& pauli_string = pauli_strings_[n];
			std::vector<uint32_t> indices;
			std::transform(pauli_string.cbegin(), pauli_string.cend(), 
					std::back_inserter(indices), [](const auto& p){ return p.first; });
			res += cps_[n]->apply(indices, vec);
		}
		return res;
	}

	/* this function makes sense only when operators are mutually commuting 
	 *
	 * */
	Eigen::VectorXcd apply_exp(cx_double t, const Eigen::VectorXcd& vec) const
	{
		assert(vec.size() == (1u << num_qubits_));

		if(cps_.size() < pauli_strings_.size())
			update_cps();

		Eigen::VectorXcd res = vec;
		for(std::size_t n = 0; n < pauli_strings_.size(); ++n)
		{
			const auto& pauli_string = pauli_strings_[n];
			std::vector<uint32_t> indices;
			std::transform(pauli_string.cbegin(), pauli_string.cend(), 
					std::back_inserter(indices), [](const auto& p){ return p.first; });
			res = cps_[n]->apply_exp(t, indices, res);
		}
		return res;
	}


};
}

class SumPauliString final
	: public Operator
{
public:
	using PauliString = std::map<uint32_t, Pauli>;

private:
	std::shared_ptr<const detail::SumPauliStringImpl> p_;
	cx_double constant_ = 1.0;

	void dagger_in_place_impl() override
	{
		constant_ = std::conj(constant_);
	}


public:
	explicit SumPauliString(const uint32_t num_qubits, std::string name = {})
		: Operator(1<<num_qubits, std::move(name)),
		p_{std::make_shared<detail::SumPauliStringImpl>(num_qubits)}
	{
	}

	explicit SumPauliString(const uint32_t num_qubits,
			const std::vector<std::map<uint32_t, Pauli>>& pauli_strings, 
			std::string name = {})
		: Operator(1u << num_qubits, std::move(name)),
		p_{std::make_shared<detail::SumPauliStringImpl>(num_qubits, pauli_strings)}
	{

	}

	explicit SumPauliString(std::shared_ptr<const detail::SumPauliStringImpl> p,
			std::string name = {}, cx_double constant = 1.0)
		: Operator(1u << p->num_qubits(), std::move(name)), p_{std::move(p)},
		constant_{constant}
	{
	}

	bool mutually_commuting() const
	{
		return p_->mutually_commuting();
	}

	std::shared_ptr<const detail::SumPauliStringImpl> get_impl() const
	{
		return p_;
	}
	/* This function might be slow. 
	 */ 
	SumPauliString& operator+=(const PauliString& str)
	{
		auto p = std::make_shared<detail::SumPauliStringImpl> (*p_);
		p->add(str);
		p_ = std::move(p);
		return *this;
	}

	SumPauliString& operator+=(PauliString&& str)
	{
		auto p = std::make_shared<detail::SumPauliStringImpl> (*p_);
		p->add(std::move(str));
		p_ = std::move(p);
		return *this;
	}

	SumPauliString(const SumPauliString& ) = default;
	SumPauliString(SumPauliString&& ) = default;

	SumPauliString& operator=(const SumPauliString& ) = delete;
	SumPauliString& operator=(SumPauliString&& ) = delete;

	std::unique_ptr<Operator> clone() const override
	{
		auto cloned = std::make_unique<SumPauliString>(*this);
		return cloned;
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return constant_*p_->apply(st);
	}
};

} //namespace yavque
