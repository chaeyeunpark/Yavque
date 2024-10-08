#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include <ranges>
#include <vector>

#include "../utils.hpp"

namespace yavque
{
enum class Pauli : char
{
	X = 'X',
	Y = 'Y',
	Z = 'Z'
};

namespace detail
{
	class CompressedPauliString
	{
	private:
		std::vector<Pauli> pstring_;
		Eigen::MatrixXcd mat_;
		mutable bool diagonalized_ = false;
		mutable Eigen::MatrixXcd evecs_;
		mutable Eigen::MatrixXcd evals_;

		static std::vector<Pauli> construct_pauli(const std::string& str)
		{
			std::vector<Pauli> res;
			for(auto c : str)
			{
				res.emplace_back(Pauli(c));
			}
			return res;
		}

		static Eigen::MatrixXcd get_pauli(Pauli p)
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
			return {};
		}

		void diagonalize() const
		{
			if(!diagonalized_)
			{
				const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(mat_);
				evecs_ = es.eigenvectors();
				evals_ = es.eigenvalues();
				diagonalized_ = true;
			}
		}

		static uint32_t change_bits(const std::vector<uint32_t>& indices,
		                            uint32_t bitstring, uint32_t bits_to_change)
		{
			for(uint32_t n = 0; n < indices.size(); ++n)
			{
				const uint32_t b = (bits_to_change >> n) & 1U;
				bitstring = (bitstring & (~(1U << indices[n]))) | (b << indices[n]);
			}
			return bitstring;
		}

		static uint32_t bits(const std::vector<uint32_t>& indices, uint32_t bitstring)
		{
			uint32_t b = 0U;

			for(uint32_t k = 0; k < indices.size(); ++k)
			{
				b |= ((bitstring >> indices[k]) & 1U) << k;
			}
			return b;
		}

		void construct_matrix()
		{
			mat_ = Eigen::MatrixXcd::Ones(1, 1);
			for(auto iter : std::ranges::reverse_view(pstring_))
			{
				mat_ = Eigen::kroneckerProduct(mat_, get_pauli(iter)).eval();
			}
		}

	public:
		explicit CompressedPauliString(const std::string& str)
			: pstring_{construct_pauli(str)}
		{
			construct_matrix();
		}

		explicit CompressedPauliString(std::vector<Pauli> pvec)
			: pstring_{std::move(pvec)}
		{
			construct_matrix();
		}

		Pauli at(uint32_t idx) const { return pstring_[idx]; }

		Eigen::VectorXcd apply(const std::vector<uint32_t>& indices,
		                       const Eigen::VectorXcd& vec) const
		{
			assert(pstring_.size() == indices.size());
			const uint32_t dim = 1U << pstring_.size();
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
				const uint32_t row = bits(indices, k);
				for(uint32_t col = 0; col < dim; ++col)
				{
					const uint32_t l = change_bits(indices, k, col);
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
			const uint32_t dim = 1U << pstring_.size();
			Eigen::VectorXcd res = Eigen::VectorXcd::Zero(vec.size());

			if(!diagonalized_)
			{
				diagonalize();
			}

			const Eigen::VectorXcd p = (t * evals_.array()).exp();
			Eigen::MatrixXcd exp_mat = evecs_ * p.asDiagonal() * evecs_.adjoint();

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
				return apply_three_qubit(vec, exp_mat,
				                         {indices[0], indices[1], indices[2]});
			}

			for(uint32_t k = 0; k < vec.size(); ++k)
			{
				cx_double v = 0.0;
				const uint32_t row = bits(indices, k);
				for(uint32_t col = 0; col < dim; ++col)
				{
					const uint32_t l = change_bits(indices, k, col);
					v += exp_mat(row, col) * vec(l);
				}
				res(k) = v;
			}
			return res;
		}
	};
} // namespace detail

class CPSFactory
{
private:
	std::map<std::string, std::weak_ptr<detail::CompressedPauliString>> map_;

	CPSFactory() = default;

public:
	CPSFactory(const CPSFactory&) = delete;
	CPSFactory& operator=(const CPSFactory&) = delete;

	CPSFactory(CPSFactory&&) = delete;
	CPSFactory& operator=(CPSFactory&&) = delete;
	~CPSFactory() = default;

	std::shared_ptr<detail::CompressedPauliString>
	get_pauli_string_for(const std::string& pstr)
	{
		auto iter = map_.find(pstr);
		if((iter == map_.end()) || iter->second.expired())
		{
			auto p = std::make_shared<detail::CompressedPauliString>(pstr);
			map_[pstr] = p;
			return p;
		}
		return std::shared_ptr<detail::CompressedPauliString>(map_[pstr]);
	}

	static CPSFactory& get_instance()
	{
		static CPSFactory instance;
		return instance;
	}
};
} // namespace yavque
