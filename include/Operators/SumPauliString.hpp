#pragma once

#include <set>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/KroneckerProduct>

#include "utilities.hpp"

#include "Operators/Operator.hpp"

namespace qunn
{
enum class Pauli: char {X='X', Y='Y', Z='Z'};

namespace detail
{

bool commute(const std::map<uint32_t, Pauli>& p1, const std::map<uint32_t, Pauli>& p2);
std::string extract_pauli_string(const std::map<uint32_t, Pauli>& pmap);

class CompressedPauliString
{
private:
	const std::vector<Pauli> pstring_;
	Eigen::MatrixXcd mat_;
	mutable bool diagonalized_ = false;
	mutable Eigen::MatrixXcd evecs_;
	mutable Eigen::MatrixXcd evals_;

	std::vector<Pauli> construct_pauli(const std::string& str)
	{
		std::vector<Pauli> res;
		for(auto c: str)
		{
			res.emplace_back(Pauli(c));
		}
		return res;
	}

	Eigen::MatrixXcd get_pauli(Pauli p) const
	{
		switch(p)
		{
		case Pauli::X:
			return pauli_x().cast<cx_double>();
		case Pauli::Y:
			return pauli_y();
		case Pauli::Z:
			return pauli_z().cast<cx_double>();
		}
		__builtin_unreachable();
		return Eigen::MatrixXcd();
	}

	void diagonalize() const
	{
		if(!diagonalized_)
		{
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(mat_);
			evecs_ = es.eigenvectors();
			evals_ = es.eigenvalues();
			diagonalized_ = true;
		}
	}

	static uint32_t change_bits(const std::vector<uint32_t>& indices, 
			uint32_t bitstring, uint32_t bits_to_change)
	{
		for(uint32_t n = 0; n <  indices.size(); ++n)
		{
			uint32_t b = (bits_to_change >> n) & 1;
			bitstring = (bitstring & (~(1 << indices[n]))) | (b << indices[n]);
		}
		return bitstring;
	}

	static uint32_t bits(const std::vector<uint32_t>& indices, 
			uint32_t bitstring)
	{
		uint32_t b = 0u;

		for(uint32_t k = 0; k < indices.size(); ++k)
		{
			b |= ((bitstring >> indices[k]) & 1) << k;
		}
		return b;
	}

	void construct_matrix()
	{
		mat_ = Eigen::MatrixXcd::Ones(1,1);
		for(auto iter = pstring_.rbegin(); iter != pstring_.rend(); ++iter)
		{
			mat_ = Eigen::KroneckerProduct(mat_, get_pauli(*iter)).eval();
		}
	}
	
public:
	explicit CompressedPauliString(const std::string& str)
		: pstring_{construct_pauli(str)}
	{
		construct_matrix();
	}

	explicit CompressedPauliString(const std::vector<Pauli>& pvec)
		: pstring_{pvec}
	{
		construct_matrix();
	}

	Pauli at(uint32_t idx) const
	{
		return pstring_[idx];
	}

	Eigen::VectorXcd apply(const std::vector<uint32_t>& indices, 
			const Eigen::VectorXcd& vec) const
	{
		assert(pstring_.size() == indices.size());
		uint32_t dim = 1u << pstring_.size();
		Eigen::VectorXcd res = Eigen::VectorXcd::Zero(vec.size());

		if(indices.size() == 1)
		{
			return apply_single_qubit(vec, mat_, indices[0]);
		}
		if(indices.size() == 2)
		{
			return apply_two_qubit(vec, mat_, {indices[0], indices[1]});
		}
		if(indices.size() == 3)
		{
			return apply_three_qubit(vec, mat_, {indices[0], indices[1], indices[2]});
		}
		
		for(uint32_t k = 0; k < vec.size(); ++k)
		{
			cx_double v = 0.0;
			uint32_t row = bits(indices, k);
			for(uint32_t col = 0; col < dim; ++col)
			{
				uint32_t l = change_bits(indices, k, col);
				v += mat_(row, col) * vec(l);
			}
			res(k) = v;
		}
		return res;
	}

	/* return exp(t*P) applied to indices */
	Eigen::VectorXcd apply_exp(cx_double t, const std::vector<uint32_t>& indices, 
			const Eigen::VectorXcd& vec) const
	{
		assert(pstring_.size() == indices.size());
		uint32_t dim = 1u << pstring_.size();
		Eigen::VectorXcd res = Eigen::VectorXcd::Zero(vec.size());

		if(!diagonalized_)
			diagonalize();

		Eigen::VectorXcd p = (t*evals_.array()).exp();
		Eigen::MatrixXcd exp_mat = evecs_*p.asDiagonal()*evecs_.adjoint();

		if(indices.size() == 1)
		{
			return apply_single_qubit(vec, exp_mat, indices[0]);
		}
		if(indices.size() == 2)
		{
			return apply_two_qubit(vec, exp_mat, {indices[0], indices[1]});
		}
		if(indices.size() == 3)
		{
			return apply_three_qubit(vec, exp_mat, {indices[0], indices[1], indices[2]});
		}
		
		for(uint32_t k = 0; k < vec.size(); ++k)
		{
			cx_double v = 0.0;
			uint32_t row = bits(indices, k);
			for(uint32_t col = 0; col < dim; ++col)
			{
				uint32_t l = change_bits(indices, k, col);
				v += exp_mat(row, col) * vec(l);
			}
			res(k) = v;
		}
		return res;
	}
};


class SumPauliStringImpl
{
public:
	using PauliString = std::map<uint32_t, Pauli>;

private:
	const uint32_t num_qubits_;
	std::vector<PauliString> pauli_strings_;
	mutable std::map<std::string, CompressedPauliString> unique_strings_;
	mutable std::size_t updated_to_ = 0;

	void update_unique_strings() const
	{
		for(std::size_t idx = updated_to_; idx < pauli_strings_.size(); ++idx)
		{
			std::string pstr = extract_pauli_string(pauli_strings_[idx]);
			unique_strings_.try_emplace(pstr, CompressedPauliString(pstr));
		}
		updated_to_ = pauli_strings_.size();
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
		if(updated_to_ != pauli_strings_.size())
			update_unique_strings();
		Eigen::VectorXcd res = Eigen::VectorXcd::Zero(vec.size());
		for(const auto& pauli_string: pauli_strings_)
		{
			auto iter = unique_strings_.find(extract_pauli_string(pauli_string));
			std::vector<uint32_t> indices;
			std::transform(pauli_string.cbegin(), pauli_string.cend(), 
					std::back_inserter(indices), [](const auto& p){ return p.first; });
			res += (iter->second).apply(indices, vec);
		}
		return res;
	}

	/* this function makes sense only when operators are mutually commuting 
	 *
	 * */
	Eigen::VectorXcd apply_exp(cx_double t, const Eigen::VectorXcd& vec) const
	{
		assert(vec.size() == (1u << num_qubits_));
		if(updated_to_ != pauli_strings_.size())
			update_unique_strings();
		Eigen::VectorXcd res = vec;
		for(const auto& pauli_string: pauli_strings_)
		{
			auto iter = unique_strings_.find(extract_pauli_string(pauli_string));
			std::vector<uint32_t> indices;
			std::transform(pauli_string.cbegin(), pauli_string.cend(), 
					std::back_inserter(indices), [](const auto& p){ return p.first; });
			res = (iter->second).apply_exp(t, indices, res);
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
	explicit SumPauliString(const uint32_t num_qubits)
		: Operator(1<<num_qubits),
		p_{std::make_shared<detail::SumPauliStringImpl>(num_qubits)}
	{
	}
	explicit SumPauliString(const uint32_t num_qubits,
			const std::vector<std::map<uint32_t, Pauli>>& pauli_strings)
		: Operator(1<<num_qubits),
		p_{std::make_shared<detail::SumPauliStringImpl>(num_qubits, pauli_strings)}
	{

	}

	explicit SumPauliString(const uint32_t num_qubits,
			const std::vector<std::map<uint32_t, Pauli>>& pauli_strings, 
			std::string name)
		: Operator(1u << num_qubits, std::move(name)),
		p_{std::make_shared<detail::SumPauliStringImpl>(num_qubits, pauli_strings)}
	{

	}

	explicit SumPauliString(std::shared_ptr<const detail::SumPauliStringImpl> p)
		: Operator(1u << p->num_qubits()), p_{std::move(p)}
	{
	}

	explicit SumPauliString(std::shared_ptr<const detail::SumPauliStringImpl> p,
			std::string name, cx_double constant = 1.0)
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
		cloned->set_name(std::string("clone of ") + name());
		return cloned;
	}

	Eigen::VectorXcd apply_right(const Eigen::VectorXcd& st) const override
	{
		return constant_*p_->apply(st);
	}
};

} //namespace qunn
