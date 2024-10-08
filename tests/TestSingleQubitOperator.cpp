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

Eigen::MatrixXcd matrix_log(const Eigen::MatrixXcd& m)
{
	const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(m);
	const Eigen::MatrixXcd& u = solver.eigenvectors();

	Eigen::VectorXcd d = solver.eigenvalues();
	d.array() = d.array().log().eval();
	return u * d.asDiagonal() * u.adjoint();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("test single qubit operator", "[single-qubit-operator]")
{
	using namespace yavque;
	constexpr uint32_t N = 10U;
	constexpr uint32_t dim = 1U << N;
	constexpr cx_double I(0, 1.0);
	std::mt19937_64 re{1557U};
	std::uniform_int_distribution<uint32_t> index_dist(0, N - 1);

	// test using sparse matrix construction
	for(uint32_t instance_idx = 0; instance_idx < 100; ++instance_idx)
	{
		auto op = random_unitary(2, re);
		auto idx = index_dist(re);
		auto m1 = SingleQubitOperator(op, N, idx);

		auto st = random_vector(dim, re);

		edp::LocalHamiltonian<cx_double> lh(N, 2);
		lh.addOneSiteTerm(idx, op.sparseView());
		auto m = edp::constructSparseMat<cx_double>(dim, lh);

		REQUIRE((m1.apply_right(st) - m * st).norm() < 1e-6);

		m1.dagger_in_place();
		REQUIRE((m1.apply_right(st) - m.adjoint() * st).norm() < 1e-6);
	}

	// test using U = e^{-I H}
	for(uint32_t instance_idx = 0; instance_idx < 100; ++instance_idx)
	{
		auto op = random_unitary(2, re);
		auto idx = index_dist(re);
		auto m1 = SingleQubitOperator(op, N, idx);

		auto m2 = SingleQubitHamEvol(
			std::make_shared<DenseHermitianMatrix>(I * matrix_log(op)), N, idx);
		m2.set_variable_value(1.0);

		auto st = random_vector(dim, re);
		REQUIRE((m1.apply_right(st) - m2.apply_right(st)).norm() < 1e-6);
		m1.dagger_in_place();
		m2.dagger_in_place();
		REQUIRE((m1.apply_right(st) - m2.apply_right(st)).norm() < 1e-6);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("test derivative of single qubit ham evol", "[single-qubit-ham-evol]")
{
	using namespace yavque;
	constexpr uint32_t N = 10;
	constexpr uint32_t dim = 1U << N;

	std::mt19937_64 re{1557U};
	// NOLINTBEGIN(misc-const-correctness)
	std::normal_distribution<double> ndist;
	std::uniform_int_distribution<uint32_t> index_dist(0, N - 1);
	// NOLINTEND(misc-const-correctness)

	SECTION("test using pauli-x")
	{
		for(uint32_t instance_idx = 0; instance_idx < 10; ++instance_idx)
		{
			auto idx = index_dist(re);
			auto m = SingleQubitHamEvol(std::make_shared<DenseHermitianMatrix>(pauli_x()),
			                            N, idx);

			const double val = ndist(re);
			m.set_variable_value(val);

			const auto st = random_vector(dim, re);
			const Eigen::VectorXcd grad1 = m.log_deriv()->apply_right(m.apply_right(st));

			// from parameter shift rule
			m.get_variable() += M_PI / 2;
			Eigen::VectorXcd grad2 = m.apply_right(st);
			m.get_variable() = val - M_PI / 2;
			grad2 -= m.apply_right(st);
			grad2 /= 2.0;

			REQUIRE((grad1 - grad2).norm() < 1e-6);
		}
	}
	SECTION("test using pauli-y")
	{
		for(uint32_t instance_idx = 0; instance_idx < 10; ++instance_idx)
		{
			auto idx = index_dist(re);
			auto m = SingleQubitHamEvol(std::make_shared<DenseHermitianMatrix>(pauli_y()),
			                            N, idx);

			const double val = ndist(re);
			m.set_variable_value(val);

			const auto st = random_vector(dim, re);
			const Eigen::VectorXcd grad1 = m.log_deriv()->apply_right(m.apply_right(st));

			// from parameter shift rule
			m.get_variable() += M_PI / 2;
			Eigen::VectorXcd grad2 = m.apply_right(st);
			m.get_variable() = val - M_PI / 2;
			grad2 -= m.apply_right(st);
			grad2 /= 2.0;

			REQUIRE((grad1 - grad2).norm() < 1e-6);
		}
	}
	SECTION("test using pauli-z")
	{
		for(uint32_t instance_idx = 0; instance_idx < 10; ++instance_idx)
		{
			auto idx = index_dist(re);
			auto m = SingleQubitHamEvol(std::make_shared<DenseHermitianMatrix>(pauli_z()),
			                            N, idx);

			const double val = ndist(re);
			m.set_variable_value(val);

			const auto st = random_vector(dim, re);
			const Eigen::VectorXcd grad1 = m.log_deriv()->apply_right(m.apply_right(st));

			// from parameter shift rule
			m.get_variable() += M_PI / 2;
			Eigen::VectorXcd grad2 = m.apply_right(st);
			m.get_variable() = val - M_PI / 2;
			grad2 -= m.apply_right(st);
			grad2 /= 2.0;

			REQUIRE((grad1 - grad2).norm() < 1e-6);
		}
	}

	SECTION("test using random hermitian matrix")
	{
		for(uint32_t instance_idx = 0; instance_idx < 100; ++instance_idx)
		{
			auto idx = index_dist(re);

			Eigen::Vector3d coeffs;
			for(uint32_t i = 0; i < 3; ++i)
			{
				coeffs(i) = ndist(re);
			}

			Eigen::MatrixXcd ham = coeffs(0) * pauli_x();
			ham += coeffs(1) * pauli_y();
			ham += coeffs(2) * pauli_z();

			auto m
				= SingleQubitHamEvol(std::make_shared<DenseHermitianMatrix>(ham), N, idx);

			const double val = ndist(re);
			m.set_variable_value(val);

			const auto st = random_vector(dim, re);
			const Eigen::VectorXcd grad1 = m.log_deriv()->apply_right(m.apply_right(st));

			// Compute the gradient from the parameter shift rule
			Eigen::VectorXcd grad2;
			{
				const double t = coeffs.norm();
				m.get_variable() += M_PI / 2 / t;
				grad2 = m.apply_right(st);
				m.get_variable() = val - M_PI / 2 / t;
				grad2 -= m.apply_right(st);

				grad2 *= t / 2.0;
			}

			REQUIRE((grad1 - grad2).norm() < 1e-6);
		}
	}
}
