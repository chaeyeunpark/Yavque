#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include "utility.hpp"
Eigen::VectorXcd product_state(uint32_t n, const Eigen::VectorXcd& s)
{
	Eigen::VectorXcd res(1);
	res(0) = 1.0;
	for(uint32_t i = 0; i < n ; i++)
	{
		res = Eigen::kroneckerProduct(res,s).eval();
	}
	return res;
}

template<typename T>
Eigen::VectorXcd apply_kronecker(uint32_t N, 
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, 
		const Eigen::VectorXcd& vec)
{
	Eigen::VectorXcd res = vec;
	for(uint32_t k = 0; k < N; ++k)
	{
		res = qunn::apply_single_qubit(res, m, k);
	}
	return res;
}

template<typename RandomEngine>
std::pair<uint32_t, uint32_t> random_connection(const int N, RandomEngine& re)
{
	std::uniform_int_distribution<uint32_t> uid1(0, N-1);
	std::uniform_int_distribution<uint32_t> uid2(0, N-2);

	auto r1 = uid1(re);
	auto r2 = uid2(re);

	if(r2 < r1)
		return std::make_pair(r1, r2);
	else
		return std::make_pair(r1, r2+1);
}


