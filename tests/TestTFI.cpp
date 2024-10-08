#include "common.hpp"

#include "yavque/Circuit.hpp"
#include "yavque/operators.hpp"
#include "yavque/utils.hpp"

#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/EDP/LocalHamiltonian.hpp"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

#include <random>

yavque::Circuit construct_diagonal_tfi(const uint32_t N)
{
	yavque::Circuit circ(1U << N);
	Eigen::VectorXd zz_all(1U << N);

	for(uint32_t n = 0; n < (1U << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; ++k)
		{
			const int z0 = 1 - 2 * static_cast<int>((n >> k) & 1U);
			const int z1 = 1 - 2 * static_cast<int>((n >> ((k + 1) % N)) & 1U);
			elt += z0 * z1;
		}
		zz_all(n) = elt;
	}

	auto zz_all_ham = yavque::DiagonalOperator(zz_all, "zz all");
	auto x_all_ham
		= yavque::SumLocalHam(N, yavque::pauli_x().cast<yavque::cx_double>(), "x all");

	for(uint32_t p = 0; p < 4; ++p)
	{
		circ.add_op_right<yavque::DiagonalHamEvol>(zz_all_ham);
		circ.add_op_right<yavque::SumLocalHamEvol>(x_all_ham);
	}
	circ.add_op_right<yavque::DiagonalHamEvol>(zz_all_ham);
	return circ;
}

auto construct_bare_tfi(const uint32_t N)
{
	yavque::Circuit circ(1U << N);
	std::vector<yavque::Variable> variables(9);

	std::vector<yavque::Hamiltonian> zz_hams;

	for(uint32_t k = 0; k < N; ++k)
	{
		edp::LocalHamiltonian<double> lh(N, 2);
		lh.addTwoSiteTerm({k, (k + 1) % N}, yavque::pauli_zz());
		zz_hams.emplace_back(edp::constructSparseMat<yavque::cx_double>(1U << N, lh));
	}
	auto x_all_ham
		= yavque::SumLocalHam(N, yavque::pauli_x().cast<yavque::cx_double>(), "x all");

	for(uint32_t p = 0; p < 4; ++p)
	{
		for(uint32_t k = 0; k < N; k++)
		{
			circ.add_op_right<yavque::HamEvol>(zz_hams[k], variables[2 * p + 0]);
		}
		circ.add_op_right<yavque::SumLocalHamEvol>(x_all_ham, variables[2 * p + 1]);
	}
	for(uint32_t k = 0; k < N; k++)
	{
		circ.add_op_right<yavque::HamEvol>(zz_hams[k], variables[8]);
	}

	return std::make_pair(std::move(circ), std::move(variables));
}

Eigen::VectorXcd analytic_twoqubit(double theta, double phi)
{
	Eigen::VectorXcd res(4);
	constexpr yavque::cx_double I(0., 1.);

	res(0) = cos(2 * phi) * exp(-I * theta) - I * sin(2 * phi) * exp(I * theta);
	res(1) = cos(2 * phi) * exp(I * theta) - I * sin(2 * phi) * exp(-I * theta);
	res(2) = cos(2 * phi) * exp(I * theta) - I * sin(2 * phi) * exp(-I * theta);
	res(3) = cos(2 * phi) * exp(-I * theta) - I * sin(2 * phi) * exp(I * theta);

	return res / 2;
}

Eigen::MatrixXcd rot_x(double phi)
{
	constexpr yavque::cx_double I(0., 1.);
	Eigen::MatrixXcd rot_x(2, 2);

	rot_x << cos(phi), -I * sin(phi), -I * sin(phi), cos(phi);

	return rot_x;
}

Eigen::MatrixXcd kron_n(const Eigen::MatrixXcd& m, uint32_t n)
{
	Eigen::MatrixXcd res(m);
	for(uint32_t k = 1; k < n; ++k)
	{
		res = Eigen::kroneckerProduct(res, m).eval();
	}
	return res;
}

Eigen::VectorXcd product_fourqubit(double theta1, double phi1, double theta2, double phi2)
{
	constexpr yavque::cx_double I(0., 1.);

	Eigen::VectorXd zz(16);
	zz << 4, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, 0, 0, 0, 4;

	Eigen::VectorXcd v = Eigen::VectorXcd::Ones(16);
	v /= 4.0;

	v.array() *= (-I * theta1 * zz.array()).exp();
	v = kron_n(rot_x(phi1), 4) * v;
	v.array() *= (-I * theta2 * zz.array()).exp();
	v = kron_n(rot_x(phi2), 4) * v;

	return v;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("test two qubit", "[tfi-twoqubit]")
{
	using namespace yavque;
	using namespace Eigen;

	std::mt19937_64 re{1557U};

	std::normal_distribution<double> ndist{};

	constexpr unsigned int N = 2;
	const double eps = 1e-6;

	Eigen::VectorXcd zz(1U << N);

	for(uint32_t n = 0; n < (1U << N); ++n)
	{
		const int z0 = 1 - 2 * static_cast<int>(n & 1U);
		const int z1 = 1 - 2 * static_cast<int>((n >> 1U) & 1U);
		zz(n) = z0 * z1;
	}

	auto zz_all_ham = yavque::DiagonalOperator(zz, "zz all");
	auto x_all_ham
		= yavque::SumLocalHam(N, yavque::pauli_x().cast<yavque::cx_double>(), "x all");

	auto circ = yavque::Circuit(1U << N);

	for(uint32_t p = 0; p < 1; ++p)
	{
		circ.add_op_right<yavque::DiagonalHamEvol>(zz_all_ham);
		circ.add_op_right<yavque::SumLocalHamEvol>(x_all_ham);
	}

	Eigen::MatrixXd ham(4, 4);
	ham << -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1;

	auto variables = circ.variables();

	Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(4);
	ini /= sqrt(4.0);

	circ.set_input(ini);

	for(uint32_t _instance = 0; _instance < 10; ++_instance)
	{
		double theta = ndist(re);
		double phi = ndist(re);
		variables[0] = theta;
		variables[1] = phi;

		for(uint32_t epoch = 0; epoch < 100; ++epoch)
		{ // learning loop
			theta = variables[0].value();
			phi = variables[1].value();

			circ.clear_evaluated();
			const Eigen::VectorXcd output = *circ.output();
			const Eigen::VectorXcd analytic = analytic_twoqubit(theta, phi);

			REQUIRE((output - analytic).norm() < 1e-6);

			// test grad
			for(auto& p : variables)
			{
				p.zero_grad();
			}
			circ.derivs();

			Eigen::MatrixXcd grads(1U << N, 2);
			grads.col(0) = *variables[0].grad();
			grads.col(1) = *variables[1].grad();
			Eigen::VectorXd egrad_circ = 2 * (output.adjoint() * ham * grads).real();

			Eigen::VectorXcd v1 = analytic_twoqubit(theta + eps, phi);
			Eigen::VectorXcd v2 = analytic_twoqubit(theta - eps, phi);

			Eigen::VectorXd egrad_num(2);
			egrad_num.coeffRef(0) = real(cx_double(v1.adjoint() * ham * v1)
			                             - cx_double(v2.adjoint() * ham * v2))
			                        / (2 * eps);

			v1 = analytic_twoqubit(theta, phi + eps);
			v2 = analytic_twoqubit(theta, phi - eps);

			egrad_num.coeffRef(1) = real(cx_double(v1.adjoint() * ham * v1)
			                             - cx_double(v2.adjoint() * ham * v2))
			                        / (2 * eps);

			REQUIRE((egrad_circ - egrad_num).norm() < 1e-6);

			for(uint32_t k = 0; k < variables.size(); ++k)
			{
				variables[k] -= 0.02 * egrad_circ(k);
			}

			// std::cout << real(cx_double(output.adjoint() * ham * output)) << std::endl;
		}
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("test four qubit", "[tfi-fourqubit]")
{
	using namespace yavque;
	using namespace Eigen;

	std::mt19937_64 re{1557U};
	std::normal_distribution<double> ndist{};

	constexpr unsigned int N = 4;
	const double eps = 1e-6;

	Eigen::VectorXd zz_all(1U << N);

	for(uint32_t n = 0; n < (1U << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; ++k)
		{
			const int z0 = 1 - 2 * static_cast<int>((n >> k) & 1U);
			const int z1 = 1 - 2 * static_cast<int>((n >> ((k + 1) % N)) & 1U);
			elt += z0 * z1;
		}
		zz_all(n) = elt;
	}

	auto zz_all_ham = yavque::DiagonalOperator(zz_all, "zz all");
	auto x_all_ham
		= yavque::SumLocalHam(N, yavque::pauli_x().cast<yavque::cx_double>(), "x all");

	auto circ = yavque::Circuit(1U << N);

	for(uint32_t p = 0; p < 2; ++p)
	{
		circ.add_op_right<yavque::DiagonalHamEvol>(zz_all_ham);
		circ.add_op_right<yavque::SumLocalHamEvol>(x_all_ham);
	}

	const Eigen::MatrixXd ham = zz_all.asDiagonal();

	auto variables = circ.variables();

	Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(16);
	ini /= 4.0;

	circ.set_input(ini);

	for(uint32_t _instance = 0; _instance < 10; ++_instance)
	{
		double theta1 = ndist(re);
		double theta2 = ndist(re);
		double phi1 = ndist(re);
		double phi2 = ndist(re);

		variables[0] = theta1;
		variables[1] = phi1;
		variables[2] = theta2;
		variables[3] = phi2;

		for(uint32_t epoch = 0; epoch < 100; ++epoch)
		{ // learning loop
			theta1 = variables[0].value();
			phi1 = variables[1].value();
			theta2 = variables[2].value();
			phi2 = variables[3].value();

			circ.clear_evaluated();
			const Eigen::VectorXcd output = *circ.output();
			const Eigen::VectorXcd analytic = product_fourqubit(theta1, phi1, theta2, phi2);

			REQUIRE((output - analytic).norm() < 1e-6);

			// test grad
			for(auto& p : variables)
			{
				p.zero_grad();
			}
			circ.derivs();
			Eigen::MatrixXcd grads(1U << N, 4);
			for(uint32_t k = 0; k < 4; k++)
			{
				grads.col(k) = *variables[k].grad();
			}
			Eigen::VectorXd egrad_circ = 2 * (output.adjoint() * ham * grads).real();

			Eigen::VectorXcd v1 = product_fourqubit(theta1 + eps, phi1, theta2, phi2);
			Eigen::VectorXcd v2 = product_fourqubit(theta1 - eps, phi1, theta2, phi2);

			Eigen::VectorXd egrad_num(4);
			egrad_num.coeffRef(0) = real(cx_double(v1.adjoint() * ham * v1)
			                             - cx_double(v2.adjoint() * ham * v2))
			                        / (2 * eps);

			v1 = product_fourqubit(theta1, phi1 + eps, theta2, phi2);
			v2 = product_fourqubit(theta1, phi1 - eps, theta2, phi2);
			egrad_num.coeffRef(1) = real(cx_double(v1.adjoint() * ham * v1)
			                             - cx_double(v2.adjoint() * ham * v2))
			                        / (2 * eps);

			v1 = product_fourqubit(theta1, phi1, theta2 + eps, phi2);
			v2 = product_fourqubit(theta1, phi1, theta2 - eps, phi2);
			egrad_num.coeffRef(2) = real(cx_double(v1.adjoint() * ham * v1)
			                             - cx_double(v2.adjoint() * ham * v2))
			                        / (2 * eps);

			v1 = product_fourqubit(theta1, phi1, theta2, phi2 + eps);
			v2 = product_fourqubit(theta1, phi1, theta2, phi2 - eps);
			egrad_num.coeffRef(3) = real(cx_double(v1.adjoint() * ham * v1)
			                             - cx_double(v2.adjoint() * ham * v2))
			                        / (2 * eps);

			REQUIRE((egrad_circ - egrad_num).norm() < 1e-6);

			for(uint32_t k = 0; k < variables.size(); ++k)
			{
				variables[k] -= 0.02 * egrad_circ(k);
			}

			// std::cout << real(cx_double(output.adjoint() * ham * output)) << std::endl;
		}
	}
}

TEST_CASE("test tfi", "[tfi]")
{
	using namespace yavque;
	using namespace Eigen;

	constexpr unsigned int N = 8;

	std::mt19937_64 re{1557U};
	std::normal_distribution<double> nd{};

	auto circ1 = construct_diagonal_tfi(N);
	auto variables1 = circ1.variables();

	auto [circ2, variables2] = construct_bare_tfi(N);

	Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(1U << N);
	ini /= sqrt(1U << N);
	circ1.set_input(ini);
	circ2.set_input(ini);

	for(uint32_t k = 0; k < 9; ++k)
	{
		const double v = nd(re);
		variables1[k] = v;
		variables2[k] = v;
	}

	for(uint32_t _instance = 0; _instance < 100; ++_instance)
	{ // instance loop
		circ1.clear_evaluated();
		circ2.clear_evaluated();

		for(uint32_t k = 0; k < 9; ++k)
		{
			variables1[k].zero_grad();
			variables2[k].zero_grad();
		}

		const Eigen::VectorXcd output1 = *circ1.output();
		const Eigen::VectorXcd output2 = *circ2.output();

		REQUIRE((output1 - output2).norm() < 1e-6);

		circ1.derivs();
		circ2.derivs();

		for(uint32_t k = 0; k < 9; ++k)
		{
			const Eigen::VectorXcd grad1 = *variables1[k].grad();
			const Eigen::VectorXcd grad2 = *variables1[k].grad();
			REQUIRE((grad1 - grad2).norm() < 1e-6);
		}

		for(uint32_t k = 0; k < 9; ++k)
		{
			variables1[k] -= 0.01;
			variables2[k] -= 0.01;
		}
	}
}
