#include <random>
#include <iostream>
#include <cmath>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "Circuit.hpp"

#include "Operators/operators.hpp"
#include "Optimizers/OptimizerFactory.hpp"

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
	const uint32_t N = 16;
	const uint32_t depth = 10;
	const double sigma = 1.0e-2;
	const double learning_rate = 1.0e-2;

	std::random_device rd;
	std::default_random_engine re{rd()};

	Circuit circ(1 << N);

	Eigen::VectorXd zz_even(1<<N);
	Eigen::VectorXd zz_odd(1<<N);

	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; k += 2)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+1)%N)) & 1);
			elt += z0*z1;
		}
		zz_even(n) = elt;
	}

	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 1; k < N; k += 2)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+1)%N)) & 1);
			elt += z0*z1;
		}
		zz_odd(n) = elt;
	}

	auto zz_even_ham = qunn::DiagonalOperator(zz_even, "zz even");
	auto zz_odd_ham = qunn::DiagonalOperator(zz_odd, "zz odd");
	auto x_all_ham = qunn::SumLocalHam(N, qunn::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(zz_even_ham));
		circ.add_op_right(std::make_unique<qunn::DiagonalHamEvol>(zz_odd_ham));
		circ.add_op_right(std::make_unique<qunn::ProductHamEvol>(x_all_ham));
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

		Eigen::VectorXd opt = -learning_rate*fisher.inverse()*egrad;

		for(uint32_t k = 0; k < parameters.size(); ++k)
		{
			parameters[k] += opt(k);
		}
	}

	return 0;
}
