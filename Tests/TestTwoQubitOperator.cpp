#include <memory>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <tbb/tbb.h>

#include <Eigen/Dense>
#include <Eigen/src/QR/HouseholderQR.h>
#include <Eigen/src/Eigenvalues/ComplexEigenSolver.h>
#include <random>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "common.hpp"
#include "operators.hpp"

#include "Operators/SingleQubitOperator.hpp"
#include "Operators/SingleQubitHamEvol.hpp"
#include "Operators/TwoQubitOperator.hpp"
#include "Circuit.hpp"

using namespace Eigen;

tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 2);

TEST_CASE("test single qubit operator", "[single-qubit-operator]")
{
	using namespace qunn;
	constexpr uint32_t N = 3;
	constexpr uint32_t dim = 1u << N;
	constexpr cx_double I(0, 1.0);
	std::random_device rd;
	std::default_random_engine re{rd()};
	std::uniform_int_distribution<uint32_t> index_dist(0, N-1);

	// test using sparse matrix construction
	for(uint32_t instance_idx = 0; instance_idx < 100; ++instance_idx)
	{
		auto op = random_unitary(4, re);
		auto i = index_dist(re);
		auto j = index_dist(re);
		while(i == j)
		{
			j = index_dist(re);
		}
		auto m1 = TwoQubitOperator(op, N, i, j);

		auto st = random_vector(dim, re);

		edp::LocalHamiltonian<cx_double> lh(N, 2);
		lh.addTwoSiteTerm({i,j}, op.sparseView());
		auto m = edp::constructSparseMat<cx_double>(dim, lh);


		REQUIRE((m1.apply_right(st) - m*st).norm() < 1e-6);

		m1.dagger_in_place();
		REQUIRE((m1.apply_right(st) - m.adjoint()*st).norm() < 1e-6);
	}
}

