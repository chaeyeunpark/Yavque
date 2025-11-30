#include "common.hpp"

#include "yavque/operators.hpp"
#include "yavque/utils.hpp"

#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/EDP/LocalHamiltonian.hpp"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

#include <memory>
#include <random>

TEST_CASE("test random ZZ", "[random-zz]")
{
	const uint32_t N = 14;
	const uint32_t n_terms = 10;

	using namespace yavque;

	std::mt19937_64 re{1557U};
	std::normal_distribution<double> ndist;

	std::vector<uint32_t> sites;

	sites.reserve(N);
	for(uint32_t k = 0; k < N; ++k)
	{
		sites.push_back(k);
	}

	for(uint32_t k = 0; k < 100; ++k) // instance
	{
		std::vector<std::pair<uint32_t, uint32_t>> interactions;
		for(uint32_t iter_term = 0; iter_term < n_terms; ++iter_term)
		{
			std::shuffle(sites.begin(), sites.end(), re);
			interactions.emplace_back(sites[0], sites[1]);
		}

		// construct diagonal
		Eigen::VectorXd ham_diag(1U << N);
		for(uint32_t n = 0; n < (1U << N); ++n)
		{
			int elt = 0;
			for(auto [i, j] : interactions)
			{
				const int z0 = 1 - 2 * static_cast<int>((n >> i) & 1U);
				const int z1 = 1 - 2 * static_cast<int>((n >> j) & 1U);
				elt += z0 * z1;
			}
			ham_diag(n) = elt;
		}
		auto diag_ham = DiagonalOperator(ham_diag);
		auto diag_ham_evol = DiagonalHamEvol(diag_ham);

		std::vector<std::map<uint32_t, Pauli>> pauli_strings;
		for(auto [i, j] : interactions)
		{
			std::map<uint32_t, Pauli> m;
			m[i] = Pauli('Z');
			m[j] = Pauli('Z');
			pauli_strings.emplace_back(std::move(m));
		}

		auto sum_pauli = SumPauliString(N, pauli_strings);
		auto sum_pauli_evol = SumPauliStringHamEvol(sum_pauli);

		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1U << N);
		ini.normalize();

		const double t = ndist(re);

		diag_ham_evol.set_variable_value(t);
		sum_pauli_evol.set_variable_value(t);

		const auto res1 = diag_ham_evol.apply_right(ini);
		const auto res2 = sum_pauli_evol.apply_right(ini);

		REQUIRE((res1 - res2).norm() < 1e-6);
	}
}

Eigen::SparseMatrix<yavque::cx_double>
single_pauli(const uint32_t N, const uint32_t idx,
             const Eigen::SparseMatrix<yavque::cx_double>& m)
{
	edp::LocalHamiltonian<yavque::cx_double> lh(N, 2);
	lh.addOneSiteTerm(idx, m);
	return edp::constructSparseMat<yavque::cx_double>(1U << N, lh);
}
Eigen::SparseMatrix<yavque::cx_double> identity(const uint32_t N)
{
	std::vector<Eigen::Triplet<yavque::cx_double>> triplets;

	triplets.reserve(1U << N);
	for(uint32_t n = 0; n < (1U << N); ++n)
	{
		triplets.emplace_back(n, n, 1.0);
	}
	Eigen::SparseMatrix<yavque::cx_double> m(1U << N, 1U << N);
	m.setFromTriplets(triplets.begin(), triplets.end());
	return m;
}

TEST_CASE("test ZXZ", "[zxz]")
{
	const uint32_t N = 10;

	using namespace yavque;

	std::mt19937_64 re{1557U};
	std::normal_distribution<double> ndist;

	std::vector<std::map<uint32_t, Pauli>> pauli_strings;

	for(uint32_t k = 0; k < N; k++)
	{
		std::map<uint32_t, Pauli> m;
		m[k] = Pauli('Z');
		m[(k + 1) % N] = Pauli('X');
		m[(k + 2) % N] = Pauli('Z');

		pauli_strings.emplace_back(std::move(m));
	}

	auto sum_pauli = SumPauliString(N, pauli_strings);
	auto sum_pauli_evol = SumPauliStringHamEvol(sum_pauli);

	Eigen::SparseMatrix<cx_double> ham(1U << N, 1U << N);
	for(uint32_t k = 0; k < N; k++)
	{
		Eigen::SparseMatrix<cx_double> term = identity(N);
		term = term * single_pauli(N, k, pauli_z().cast<cx_double>());
		term = term * single_pauli(N, (k + 1) % N, pauli_x().cast<cx_double>());
		term = term * single_pauli(N, (k + 2) % N, pauli_z().cast<cx_double>());

		ham += term;
	}

	auto ham_full = Hamiltonian(ham);
	auto ham_full_evol = HamEvol(ham_full);

	const double t = ndist(re);

	ham_full_evol.set_variable_value(t);
	sum_pauli_evol.set_variable_value(t);

	for(uint32_t k = 0; k < 100; ++k) // instance
	{
		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1U << N);
		ini.normalize();

		const auto res1 = ham_full_evol.apply_right(ini);
		const auto res2 = sum_pauli_evol.apply_right(ini);

		REQUIRE((res1 - res2).norm() < 1e-6);
	}
}
