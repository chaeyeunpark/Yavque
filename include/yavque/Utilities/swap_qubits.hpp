#pragma once
#include <Eigen/Dense>

#include "./bit_operations.hpp"
namespace yavque
{
namespace detail
{
template<typename T>
void swap_qubits(const uint32_t num_qubits, const T* idata, T* odata, 
		uint32_t i, uint32_t j)
{
	for(uint32_t n = 0; n < (1u << num_qubits); ++n)
	{
		odata[bitswap(n, i, j)] = idata[n];
	}
}
}
}
