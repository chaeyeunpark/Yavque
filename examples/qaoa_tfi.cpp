#include <random>
#include <iostream>
#include <cmath>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "Circuit.hpp"

#include "Operators/operators.hpp"
#include "Optimizers/OptimizerFactory.hpp"

Eigen::SparseMatrix<double> zz_even(const uint32_t N)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t k = 0; k < N; k += 2)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(k, (k+1) % N), qunn::pauli_zz());
	}
	return edp::constructSparseMat<double>(1 << N, ham_ct);
}

Eigen::SparseMatrix<double> zz_odd(const uint32_t N)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t k = 1; k < N; k += 2)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(k, (k+1) % N), qunn::pauli_zz());
	}
	return edp::constructSparseMat<double>(1 << N, ham_ct);
}

Eigen::SparseMatrix<double> x_all(const uint32_t N)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t k = 0; k < N; ++k)
	{
		ham_ct.addOneSiteTerm(k, qunn::pauli_x());
	}
	return edp::constructSparseMat<double>(1 << N, ham_ct);
}

Eigen::SparseMatrix<double> tfi_ham(const uint32_t N, double h)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t k = 0; k < N; ++k)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(k, (k+1) % N), qunn::pauli_zz());
		ham_ct.addOneSiteTerm(k, h*qunn::pauli_x());
	}
	return -edp::constructSparseMat<double>(1 << N, ham_ct);
}

int main()
{
	using namespace qunn;
	using std::sqrt;
	const uint32_t N = 10;
	const uint32_t depth = 6;
	const double sigma = 1.0e-2;
	const double learning_rate = 1.0e-2;

	std::random_device rd;
	std::default_random_engine re{rd()};

	Circuit circ(1 << N);

	qunn::Hamiltonian ham_zz_even(zz_even(N).cast<cx_double>());
	qunn::Hamiltonian ham_zz_odd(zz_odd(N).cast<cx_double>());
	qunn::Hamiltonian ham_x_all(x_all(N).cast<cx_double>());


	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<HamEvol>(ham_zz_even));
		circ.add_op_right(std::make_unique<HamEvol>(ham_zz_odd));
		circ.add_op_right(std::make_unique<HamEvol>(ham_x_all));
	}

	auto parameters = circ.parameters();

	std::normal_distribution<double> ndist(0., sigma);
	for(auto& p: parameters)
	{
		p = ndist(re);
	}

	Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(1 << N);
	ini /= sqrt(1 << N);
	const auto ham = tfi_ham(N, 0.5);
	
	/*

	auto optimizer = OptimizerFactory::getInstance().createOptimizer(
			nlohmann::json{{"name", "Adam"}, {"alpha", 2e-3}});
	*/

	circ.set_input(ini);

	for(uint32_t epoch = 0; epoch < 1000; ++epoch)
	{
		circ.clear_evaluated();
		Eigen::VectorXcd output = *circ.output();
		for(auto& p: parameters)
		{
			p.zero_grad();
		}

		circ.derivs();

		Eigen::MatrixXcd grads(1 << N, parameters.size());

		for(uint32_t k = 0; k < parameters.size(); ++k)
		{
			grads.col(k) = *parameters[k].grad();
		}

		Eigen::MatrixXd fisher = (grads.adjoint()*grads).real();
		fisher += 1e-3*Eigen::MatrixXd::Identity(parameters.size(), parameters.size());

		Eigen::VectorXd egrad = (output.transpose()*ham*grads).real();
		double energy = real(cx_double(output.transpose()*ham*output));

		std::cout << energy << "\t" << egrad.norm() << "\t" << output.norm() << std::endl;

		//Eigen::VectorXd opt = optimizer->getUpdate(egrad);
		Eigen::VectorXd opt = -learning_rate*fisher.inverse()*egrad;

		for(uint32_t k = 0; k < parameters.size(); ++k)
		{
			parameters[k] += opt(k);
		}
	}

	return 0;
}
