#include "common.hpp"

#include "yavque/Operators/SumLocalHamEvol.hpp"
#include "yavque/utils.hpp"

#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/EDP/LocalHamiltonian.hpp"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

#include <random>

TEST_CASE("test sum of local hamiltonian", "[sum-local]")
{
	using namespace yavque;
	using namespace Eigen;

	constexpr unsigned int N = 8;
	constexpr cx_double I(0., 1.);

	std::mt19937_64 re{1557U};
	std::normal_distribution<double> nd;

	const SparseMatrix<cx_double> m = pauli_x().cast<cx_double>();
	const SumLocalHam ham(N, m);
	SumLocalHamEvol ham_evol(ham);
	auto var = ham_evol.get_variable();

	for(uint32_t k = 0; k < 100; ++k) // instance for loop
	{
		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1U << N);
		ini.normalize();

		const double t = nd(re);
		var = t;
		const VectorXcd out_test = ham_evol * ini;

		const MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) - I * sin(t) * m);
		const VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}

	ham_evol.dagger_in_place();
	for(uint32_t k = 0; k < 100; ++k) // instance for loop
	{
		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1U << N);
		ini.normalize();

		const double t = nd(re);
		var = t;
		const VectorXcd out_test = ham_evol * ini;

		const MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) + I * sin(t) * m);
		const VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}
}
