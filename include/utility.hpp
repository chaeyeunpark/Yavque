#pragma once
#include <complex>
#include <memory>
#include <vector>

#include <Eigen/Dense>


#include "BitOperations.h"

namespace qunn
{
using cx_double = std::complex<double>;

/* apply single qubit gate m to pos */
template<typename T>
Eigen::VectorXcd apply_single_qubit(const Eigen::VectorXcd& vec,
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, uint32_t pos)
{
	Eigen::VectorXcd res(vec.size());
	for(uint32_t k = 0; k < vec.size(); ++k)
	{
		uint32_t k0 = (k >> pos) & 1u;
		uint32_t i0 = k & (~(1u << pos));
		uint32_t i1 = k | (1u << pos);
		res(k) = m(k0, 0) * vec(i0) + m(k0, 1) * vec(i1);
	}
	return res;
}

/* apply two qubit gate to sites */
template<typename T>
Eigen::VectorXcd apply_two_qubit(const Eigen::VectorXcd& vec,
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& unitary,
		const std::vector<uint32_t>& sites)
{
	Eigen::VectorXcd res = Eigen::VectorXcd::Zero(vec.size());
	for(int i = 0; i < (vec.size() >> sites.size()); i++)
	{
		//parallelize!
		for(int j = 0; j < (1 << sites.size()); j++)
		{
			unsigned int index = (i << sites.size()) | j;
			for(uint32_t site_idx = 0; site_idx < sites.size(); ++site_idx)
			{
				bitswap(index, site_idx, sites[site_idx]);
			}
			auto c = unitary.col(j);
			for(int k = 0; k < 4; k++)
			{
				unsigned int index_to = (i << 2) | k;
				for(uint32_t site_idx = 0; site_idx < sites.size(); ++site_idx)
				{
					bitswap(index_to, site_idx, sites[site_idx]);
				}
				res(index_to) += c(k)*vec(index);
			}
		}
	}
	return res;
}
} //namespace qunn
