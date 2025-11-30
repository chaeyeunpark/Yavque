#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include "yavque/utils.hpp"

inline Eigen::VectorXcd product_state(uint32_t n, const Eigen::VectorXcd& s)
{
	Eigen::VectorXcd res(1);
	res(0) = 1.0;
	for(uint32_t i = 0; i < n; i++)
	{
		res = Eigen::kroneckerProduct(res, s).eval();
	}
	return res;
}

template<typename T>
Eigen::VectorXcd
apply_kronecker(uint32_t N, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
                const Eigen::VectorXcd& vec)
{
	Eigen::VectorXcd res = vec;
	for(uint32_t k = 0; k < N; ++k)
	{
		res = yavque::apply_single_qubit(res, m, k);
	}
	return res;
}

template<typename RandomEngine>
std::pair<uint32_t, uint32_t> random_connection(const int N, RandomEngine& re)
{
	// Somehow clang-tidy wants random distributions to be const, which is impossible
	// since operator() of those structures is non-const. Ignore them until fixed.
	// NOLINTBEGIN(misc-const-correctness)
	std::uniform_int_distribution<uint32_t> uid1(0, N - 1);
	std::uniform_int_distribution<uint32_t> uid2(0, N - 2);
	// NOLINTEND(misc-const-correctness)

	auto r1 = uid1(re);
	auto r2 = uid2(re);

	if(r2 < r1)
	{
		return std::make_pair(r1, r2);
	}
	return std::make_pair(r1, r2 + 1);
}

template<typename RandomEngine>
Eigen::MatrixXcd random_unitary(uint32_t dim, RandomEngine& re)
{
	constexpr yavque::cx_double I(0.0, 1.0);
	// This cannot be const, but clang-tidy wants. Just ignore it at this moment.
	// NOLINTNEXTLINE(misc-const-correctness)
	std::normal_distribution<double> ndist{};

	Eigen::MatrixXcd m(dim, dim);
	for(uint32_t i = 0; i < dim; ++i)
	{
		for(uint32_t j = 0; j < dim; ++j)
		{
			m(i, j) = ndist(re) + I * ndist(re);
		}
	}
	const Eigen::HouseholderQR<Eigen::MatrixXcd> qr(m);
	return qr.householderQ();
}

template<typename RandomEngine>
Eigen::VectorXcd random_vector(uint32_t dim, RandomEngine& re)
{
	constexpr yavque::cx_double I(0.0, 1.0);
	// NOLINTNEXTLINE(misc-const-correctness)
	std::normal_distribution<double> ndist{};

	Eigen::VectorXcd res(dim);
	for(uint32_t k = 0; k < dim; ++k)
	{
		res(k) = ndist(re) + I * ndist(re);
	}
	res.normalize();
	return res;
}
