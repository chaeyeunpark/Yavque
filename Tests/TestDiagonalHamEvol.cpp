#define CATCH_CONFIG_MAIN
#include <Eigen/Dense>
#include <catch.hpp>
#include <memory>
#include <random>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "yavque/Operators/DiagonalHamEvol.hpp"
#include "yavque/utils.hpp"

#include "common.hpp"

TEST_CASE("test random diagonal", "[random-diagonal]")
{
	constexpr uint32_t N = 8;
	constexpr yavque::cx_double I(0., 1.);
	std::random_device rd;
	std::default_random_engine re{rd()};

	std::normal_distribution<> nd;

	Eigen::VectorXd ham = Eigen::VectorXd::Random(1 << N);
	yavque::DiagonalOperator diag_op(ham);
	yavque::DiagonalHamEvol diag_ham_evol(diag_op);

	for(uint32_t k = 0; k < 100; ++k) // instance
	{
		Eigen::VectorXcd vec = Eigen::VectorXcd::Random(1 << N);
		vec.normalize();

		double t = nd(re);
		diag_ham_evol.set_variable_value(t);

		Eigen::VectorXcd out1 = diag_ham_evol.apply_right(vec);
		Eigen::VectorXcd out2 = exp(-I * ham.array() * t) * vec.array();

		REQUIRE((out1 - out2).norm() < 1e-6);

		Eigen::VectorXcd grad1 = diag_ham_evol.log_deriv()->apply_right(out1);
		Eigen::VectorXcd grad2 = -I * ham.cwiseProduct(out1);
		REQUIRE((grad1 - grad2).norm() < 1e-6);
	}

	diag_ham_evol.dagger_in_place();
	for(uint32_t k = 0; k < 100; ++k) // instance
	{
		Eigen::VectorXcd vec = Eigen::VectorXcd::Random(1 << N);
		vec.normalize();

		double t = nd(re);
		diag_ham_evol.set_variable_value(t);

		Eigen::VectorXcd out1 = diag_ham_evol.apply_right(vec);
		Eigen::VectorXcd out2 = exp(I * ham.array() * t) * vec.array();

		REQUIRE((out1 - out2).norm() < 1e-6);
		Eigen::VectorXcd grad1 = diag_ham_evol.log_deriv()->apply_right(out1);
		Eigen::VectorXcd grad2 = I * ham.cwiseProduct(out1);
		REQUIRE((grad1 - grad2).norm() < 1e-6);
	}
}

TEST_CASE("test basic operations", "[basic-operation]")
{
	constexpr uint32_t N = 8;
	std::random_device rd;
	std::default_random_engine re{rd()};

	std::normal_distribution<> nd;

	Eigen::VectorXd ham = Eigen::VectorXd::Random(1 << N);
	yavque::DiagonalOperator diag_op(ham);
	yavque::DiagonalHamEvol diag_ham_evol(diag_op);

	diag_ham_evol.set_variable_value(1.0);

	auto clonned = diag_ham_evol.clone();

	std::cout << clonned->desc() << std::endl;
	auto* p = dynamic_cast<yavque::Univariate*>(clonned.get());
	p->set_variable_value(-1.0);

	REQUIRE(diag_ham_evol.get_variable_value() == 1.0);
}
