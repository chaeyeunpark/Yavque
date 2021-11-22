#include "PauliHamiltonian.hpp"
#include "example_utils.hpp"
#include "yavque.hpp"

#include <EDP/ConstructSparseMat.hpp>

#include <filesystem>
#include <random>

/*
 * This program solves the chemical Hamiltonian using the hardware efficient Ansatz.
 */
int main(int argc, char* argv[])
{
	using yavque::cx_double, yavque::pauli_x, yavque::pauli_y, yavque::pauli_z,
		yavque::DenseHermitianMatrix, yavque::SingleQubitHamEvol,
		yavque::TwoQubitOperator, yavque::OptimizerFactory;
	namespace fs = std::filesystem;

	if(argc != 3)
	{
		printf("Usage: %s [data_file] [depth]\n", argv[0]);
		return 1;
	}

	auto ham = PauliHamiltonian::fromFile(fs::path(argv[1]));

	const uint32_t N = ham.getN();
	const uint32_t dim = 1U << N;
	const uint32_t total_epochs = 1000;
	const double sigma = 1.0e-3;
	const auto depth = parse_int<uint32_t>(argv[2]);

	std::random_device rd;
	std::default_random_engine re{rd()};
	auto ham_mat = edp::constructSparseMat<cx_double>(1U << N, ham);

	// construct the hardward efficient Ansatz
	yavque::Circuit circuit(1U << N);

	auto pauli_x_ham = std::make_shared<DenseHermitianMatrix>(pauli_x());
	auto pauli_y_ham = std::make_shared<DenseHermitianMatrix>(pauli_y());
	auto pauli_z_ham = std::make_shared<DenseHermitianMatrix>(pauli_z());

	Eigen::MatrixXcd cz = Eigen::MatrixXcd::Zero(4, 4);
	cz(0, 0) = 1.0;
	cz(1, 1) = 1.0;
	cz(2, 2) = 1.0;
	cz(3, 3) = -1.0;

	for(uint32_t i = 0; i < depth - 1; ++i)
	{
		for(uint32_t k = 0; k < N; ++k)
		{
			circuit.add_op_right(std::make_unique<SingleQubitHamEvol>(pauli_y_ham, N, k));
		}
		for(uint32_t k = 0; k < N; ++k)
		{
			circuit.add_op_right(std::make_unique<SingleQubitHamEvol>(pauli_x_ham, N, k));
		}
		for(uint32_t k = 0; k < N - 1; ++k)
		{
			circuit.add_op_right(std::make_unique<TwoQubitOperator>(cz, N, k, k + 1));
		}
	}
	for(uint32_t k = 0; k < N; ++k)
	{
		circuit.add_op_right(std::make_unique<SingleQubitHamEvol>(pauli_y_ham, N, k));
	}
	for(uint32_t k = 0; k < N; ++k)
	{
		circuit.add_op_right(std::make_unique<SingleQubitHamEvol>(pauli_x_ham, N, k));
	}

	std::unique_ptr<yavque::Optimizer> optimizer = nullptr;
	try
	{
		optimizer = OptimizerFactory::getInstance().createOptimizer(
			nlohmann::json{{"name", "Adam"}, {"alpha", 1.0e-2}});
	}
	catch(std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}

	// initial state is |0\rangle^{\otimes N}
	Eigen::VectorXcd ini = Eigen::VectorXcd::Zero(dim);
	ini(0) = 1.0;

	circuit.set_input(ini);

	// initialize paramerters
	auto variables = circuit.variables();
	std::normal_distribution<double> ndist(0, sigma);
	for(auto& param : variables)
	{
		param = ndist(re);
	}

	for(uint32_t epoch = 0; epoch < total_epochs; ++epoch)
	{
		circuit.clear_evaluated();
		const auto [energy, grad] = value_and_grad(ham_mat, circuit);
		const auto update = optimizer->getUpdate(grad);

		for(uint32_t i = 0; i < variables.size(); ++i)
		{
			variables[i] += update(i);
		}

		printf("%d\t%f\n", epoch, energy);
	}

	return 0;
}
