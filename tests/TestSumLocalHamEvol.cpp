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

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;

	SparseMatrix<cx_double> m = pauli_x().cast<cx_double>();
	SumLocalHam ham(N, m);
	SumLocalHamEvol ham_evol(ham);
	auto var = ham_evol.get_variable();

	for(uint32_t k = 0; k < 100; ++k) // instance for loop
	{
		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1 << N);
		ini.normalize();

		double t = nd(re);
		var = t;
		VectorXcd out_test = ham_evol * ini;

		MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) - I * sin(t) * m);
		VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}

	ham_evol.dagger_in_place();
	for(uint32_t k = 0; k < 100; ++k) // instance for loop
	{
		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1 << N);
		ini.normalize();

		double t = nd(re);
		var = t;
		VectorXcd out_test = ham_evol * ini;

		MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) + I * sin(t) * m);
		VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}
}
