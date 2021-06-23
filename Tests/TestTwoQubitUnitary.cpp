#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <random>
#include <Eigen/Dense>

#include "EDP/LocalHamiltonian.hpp"
#include "EDP/ConstructSparseMat.hpp"

#include "Operators/SumLocalHamEvol.hpp"
#include "Circuit.hpp"
#include "utilities.hpp"

#include "common.hpp"

tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 2);

TEST_CASE("test two qubit", "[tfi-twoqubit]") {
	using namespace qunn;
	using namespace Eigen;

	std::random_device rd;
	std::default_random_engine re{rd()};

	std::normal_distribution<> ndist(0., 1.);

	constexpr unsigned int N = 14;
	const double eps = 1e-6;

	Eigen::VectorXcd zz(1<<N);

	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int z0 = 1-2*((n >> 0) & 1);
		int z1 = 1-2*((n >> 1) & 1);
		zz(n) = z0*z1;
	}
	
	auto zz_all_ham = qunn::DiagonalOperator(zz, "zz all");
	auto x_all_ham = qunn::SumLocalHam(N, qunn::pauli_x().cast<qunn::cx_double>(), "x all");

	auto circ = qunn::Circuit(1 << N);

	for(uint32_t p = 0; p < 1; ++p)
	{
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(zz_all_ham));
		circ.add_op_right(std::make_unique<qunn::SumLocalHamEvol>(x_all_ham));
	}

	Eigen::MatrixXd ham(4,4);
	ham << -1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, -1;

	auto parameters = circ.parameters();

	Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(4);
	ini /= sqrt(4.0);

	circ.set_input(ini);

	for(uint32_t _instance = 0; _instance < 10; ++_instance)
	{
		double theta = ndist(re);
		double phi = ndist(re);
		parameters[0] = theta;
		parameters[1] = phi;

		for(uint32_t epoch = 0; epoch < 100; ++epoch)
		{ //learning loop
			theta = parameters[0].value();
			phi = parameters[1].value();

			circ.clear_evaluated();
			Eigen::VectorXcd output = *circ.output();
			Eigen::VectorXcd analytic = analytic_twoqubit(theta, phi);

			REQUIRE((output - analytic).norm() < 1e-6);


			//test grad
			for(auto& p: parameters)
			{
				p.zero_grad();
			}
			circ.derivs();
			Eigen::MatrixXcd grads(1<<N, 2);
			grads.col(0) = *parameters[0].grad();
			grads.col(1) = *parameters[1].grad();
			Eigen::VectorXd egrad_circ = 2*(output.adjoint()*ham*grads).real();

			Eigen::VectorXcd v1 = analytic_twoqubit(theta + eps, phi);
			Eigen::VectorXcd v2 = analytic_twoqubit(theta - eps, phi);
			
			Eigen::VectorXd egrad_num(2);
			egrad_num.coeffRef(0) = 
				real(cx_double(v1.adjoint()*ham*v1)  - cx_double(v2.adjoint()*ham*v2))/(2*eps);

			v1 = analytic_twoqubit(theta, phi + eps);
			v2 = analytic_twoqubit(theta, phi - eps);

			egrad_num.coeffRef(1) = 
				real(cx_double(v1.adjoint()*ham*v1)  - cx_double(v2.adjoint()*ham*v2))/(2*eps);

			REQUIRE((egrad_circ - egrad_num).norm() < 1e-6);

			for(uint32_t k = 0; k < parameters.size(); ++k)
			{
				parameters[k] -= 0.02*egrad_circ(k);
			}

			std::cout << real(cx_double(output.adjoint()*ham*output)) << std::endl;
		}
	}
}

