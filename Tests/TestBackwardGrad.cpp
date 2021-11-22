#include <random>
#include <sstream>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <tbb/tbb.h>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "yavque/Circuit.hpp"
#include "yavque/backward_grad.hpp"
#include "yavque/operators.hpp"

tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 2);

Eigen::Matrix2cd hadamard()
{
	using std::sqrt;
	Eigen::Matrix2cd h;
	h << 1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 1.0 / sqrt(2.0), -1.0 / sqrt(2.0);

	return h;
}

Eigen::Matrix4cd cnot()
{
	using std::sqrt;
	Eigen::Matrix4cd m;
	m.setZero();

	m(0, 0) = 1.0;
	m(1, 1) = 1.0;
	m(2, 3) = 1.0;
	m(3, 2) = 1.0;

	return m;
}

enum class Gate
{
	RotX = 0,
	RotY = 1,
	RotZ = 2, // rotations e^{-I \theta \sigma}
	Hadamard = 3,
	CNOT = 4
};

Eigen::SparseMatrix<double> tfi_ham(uint32_t N, double h)
{
	edp::LocalHamiltonian<double> lh(N, 2);
	for(uint32_t k = 0; k < N; ++k)
	{
		lh.addTwoSiteTerm({k, (k + 1) % N}, -yavque::pauli_zz());
		lh.addOneSiteTerm(k, -h * yavque::pauli_x());
	}

	return edp::constructSparseMat<double>(1u << N, lh);
}

TEST_CASE("Test gradients using a random circuit")
{
	using namespace yavque;
	constexpr uint32_t N = 14;
	constexpr uint32_t dim = 1 << N; // dimension of the total Hilbert space

	const uint32_t depth = 40;
	std::uniform_int_distribution<uint32_t> gate_dist(0, 4);
	std::uniform_int_distribution<uint32_t> qidx_dist(0, N - 1);

	std::random_device rd;
	std::default_random_engine re{rd()};

	for(uint32_t instance_idx = 0; instance_idx < 10; ++instance_idx)
	{

		// construct random circuit
		Circuit circuit(dim);
		auto pauli_x_ham = std::make_shared<DenseHermitianMatrix>(pauli_x());
		auto pauli_y_ham = std::make_shared<DenseHermitianMatrix>(pauli_y());
		auto pauli_z_ham = std::make_shared<DenseHermitianMatrix>(pauli_z());

		for(uint32_t k = 0; k < depth; ++k)
		{
			switch(static_cast<Gate>(gate_dist(re)))
			{
			case Gate::RotX:
			{
				auto idx = qidx_dist(re);
				circuit.add_op_right(
					std::make_unique<SingleQubitHamEvol>(pauli_x_ham, N, idx));
			}
			break;
			case Gate::RotY:
			{
				auto idx = qidx_dist(re);
				circuit.add_op_right(
					std::make_unique<SingleQubitHamEvol>(pauli_y_ham, N, idx));
			}
			break;
			case Gate::RotZ:
			{
				auto idx = qidx_dist(re);
				circuit.add_op_right(
					std::make_unique<SingleQubitHamEvol>(pauli_z_ham, N, idx));
			}
			break;
			case Gate::Hadamard:
			{
				auto idx = qidx_dist(re);
				circuit.add_op_right(
					std::make_unique<SingleQubitOperator>(hadamard(), N, idx));
			}
			break;
			case Gate::CNOT:
			{
				auto i = qidx_dist(re);
				auto j = qidx_dist(re);
				while(j == i)
				{
					j = qidx_dist(re);
				}
				circuit.add_op_right(std::make_unique<TwoQubitOperator>(cnot(), N, i, j));
			}
			break;
			}
		}

		auto variables = circuit.variables();
		std::normal_distribution<double> ndist;
		for(auto& param : variables)
		{
			param = ndist(re);
		}

		// set initial state |0\rangle^{\otimes N}
		Eigen::VectorXd ini = Eigen::VectorXd::Zero(dim);
		ini(0) = 1.0;
		circuit.set_input(ini);

		auto ham = tfi_ham(N, 1.0);

		circuit.derivs();
		Eigen::VectorXcd output = *circuit.output();
		Eigen::VectorXd egrad(variables.size());
		for(uint32_t k = 0; k < variables.size(); ++k)
		{
			Eigen::VectorXcd grad = *variables[k].grad();
			egrad(k) = 2 * real(cx_double(grad.adjoint() * ham * output));
		}

		const auto [value, egrad2] = value_and_grad(ham.cast<cx_double>(), circuit);

		REQUIRE((egrad - egrad2).norm() < 1e-6);
	}
}
