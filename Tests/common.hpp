#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

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

