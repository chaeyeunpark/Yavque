#include "common.hpp"

#include "yavque/Circuit.hpp"
#include "yavque/operators.hpp"

#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/EDP/LocalHamiltonian.hpp"

#include <catch2/catch_all.hpp>
#include <tbb/tbb.h>

#include <Eigen/Dense>
#include <Eigen/src/Eigenvalues/ComplexEigenSolver.h>
#include <Eigen/src/QR/HouseholderQR.h>

#include <memory>
#include <random>

using namespace Eigen;

TEST_CASE("test two qubit operator", "[two-qubit-operator]")
{
	using namespace yavque;
	constexpr uint32_t N = 3;
	constexpr uint32_t dim = 1U << N;

	std::mt19937_64 re{1557U};
	std::uniform_int_distribution<uint32_t> index_dist(0, N - 1);

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
		lh.addTwoSiteTerm({i, j}, op.sparseView());
		auto m = edp::constructSparseMat<cx_double>(dim, lh);

		REQUIRE((m1.apply_right(st) - m * st).norm() < 1e-6);

		m1.dagger_in_place();
		REQUIRE((m1.apply_right(st) - m.adjoint() * st).norm() < 1e-6);
	}
}
