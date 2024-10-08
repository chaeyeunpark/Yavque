#include "common.hpp"

#include "yavque/Operators/DiagonalHamEvol.hpp"
#include "yavque/utils.hpp"

#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/EDP/LocalHamiltonian.hpp"

#include <catch2/catch_all.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <memory>
#include <random>

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("test random diagonal", "[random-diagonal]")
{
	constexpr uint32_t N = 8;
	constexpr yavque::cx_double I(0., 1.);
	std::mt19937_64 re{1557U};

	// NOLINTNEXTLINE(misc-const-correctness)
	std::normal_distribution<double> nd{};

	Eigen::VectorXd ham = Eigen::VectorXd::Random(1U << N); // we need a seperate function for generating this
	const yavque::DiagonalOperator diag_op(ham);
	yavque::DiagonalHamEvol diag_ham_evol(diag_op);

	for(uint32_t k = 0; k < 100; ++k) // instance
	{
		Eigen::VectorXcd vec = Eigen::VectorXcd::Random(1U << N);
		vec.normalize();

		const double t = nd(re);
		diag_ham_evol.set_variable_value(t);

		const Eigen::VectorXcd out1 = diag_ham_evol.apply_right(vec);
		const Eigen::VectorXcd out2 = exp(-I * ham.array() * t) * vec.array();

		REQUIRE((out1 - out2).norm() < 1e-6);

		const Eigen::VectorXcd grad1 = diag_ham_evol.log_deriv()->apply_right(out1);
		const Eigen::VectorXcd grad2 = -I * ham.cwiseProduct(out1);
		REQUIRE((grad1 - grad2).norm() < 1e-6);
	}

	diag_ham_evol.dagger_in_place();
	for(uint32_t k = 0; k < 100; ++k) // instance
	{
		Eigen::VectorXcd vec = Eigen::VectorXcd::Random(1U << N);
		vec.normalize();

		const double t = nd(re);
		diag_ham_evol.set_variable_value(t);

		const Eigen::VectorXcd out1 = diag_ham_evol.apply_right(vec);
		const Eigen::VectorXcd out2 = exp(I * ham.array() * t) * vec.array();

		REQUIRE((out1 - out2).norm() < 1e-6);

		const Eigen::VectorXcd grad1 = diag_ham_evol.log_deriv()->apply_right(out1);
		const Eigen::VectorXcd grad2 = I * ham.cwiseProduct(out1);
		REQUIRE((grad1 - grad2).norm() < 1e-6);
	}
}

TEST_CASE("test basic operations", "[basic-operation]")
{
	constexpr uint32_t N = 8;
	// NOLINTNEXTLINE(misc-const-correctness)
	std::normal_distribution<double> nd{};

	const Eigen::VectorXd ham = Eigen::VectorXd::Random(1U << N);
	const yavque::DiagonalOperator diag_op(ham);
	yavque::DiagonalHamEvol diag_ham_evol(diag_op);

	diag_ham_evol.set_variable_value(1.0);

	auto clonned = diag_ham_evol.clone();

	// std::cout << clonned->desc() << std::endl;
	auto* p = dynamic_cast<yavque::Univariate*>(clonned.get());
	p->set_variable_value(-1.0);

	REQUIRE(diag_ham_evol.get_variable_value() == 1.0);
}
