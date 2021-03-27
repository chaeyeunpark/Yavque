#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <random>
#include <Eigen/Dense>

#include "EDP/LocalHamiltonian.hpp"
#include "EDP/ConstructSparseMat.hpp"

#include "Operators/ProductHamEvol.hpp"
#include "Operators/utils.hpp"
#include "utils.hpp"

#include "common.hpp"

TEST_CASE("test sum of local hamiltonian", "[sum-local]") {
	using namespace qunn;
	using namespace Eigen;

	constexpr unsigned int N = 8;
	constexpr cx_double I(0., 1.);


	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;


	SparseMatrix<cx_double> m = pauli_x().cast<cx_double>();
	SumLocalHam ham(N, m);
	ProductHamEvol ham_evol(ham);
	auto var = ham_evol.parameter();
	
	for(uint32_t k = 0; k < 100; ++k) //instance for loop
	{
		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1<<N);
		ini.normalize();

		double t = nd(re);
		var = t;
		VectorXcd out_test = ham_evol*ini;

		MatrixXcd mevol = (cos(t)*MatrixXcd::Identity(2,2) - I*sin(t)*m);
		VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}

	ham_evol.dagger_in_place();
	for(uint32_t k = 0; k < 100; ++k) //instance for loop
	{
		Eigen::VectorXcd ini = Eigen::VectorXcd::Random(1<<N);
		ini.normalize();

		double t = nd(re);
		var = t;
		VectorXcd out_test = ham_evol*ini;

		MatrixXcd mevol = (cos(t)*MatrixXcd::Identity(2,2) + I*sin(t)*m);
		VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}
}
