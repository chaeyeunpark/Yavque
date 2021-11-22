#include "yavque.hpp"

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include <cmath>
#include <iostream>
#include <random>

Eigen::SparseMatrix<double> tfi_ham(const uint32_t N, double h)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t k = 0; k < N; ++k)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(k, (k + 1) % N), yavque::pauli_zz());
		ham_ct.addOneSiteTerm(k, h * yavque::pauli_x());
	}
	return -edp::constructSparseMat<double>(1U << N, ham_ct);
}

int main()
{
	using std::sqrt;
	using yavque::Circuit, yavque::cx_double, yavque::Pauli,
		yavque::SumPauliStringHamEvol, yavque::SumLocalHamEvol;

	const uint32_t N = 12;
	const uint32_t depth = 6;
	const uint32_t total_epochs = 1000;

	const double sigma = 1.0e-3;
	const double learning_rate = 1.0e-2;
	const double lambda = 1.0e-3;

	std::random_device rd;
	std::default_random_engine re{rd()};

	Circuit circ(1U << N);

	yavque::SumPauliString zz_even(N);
	for(uint32_t n = 0; n < N; n += 2)
	{
		zz_even += {{n, Pauli('Z')}, {n + 1, Pauli('Z')}};
	}
	yavque::SumPauliString zz_odd(N);
	for(uint32_t n = 1; n < N; n += 2)
	{
		zz_odd += {{n, Pauli('Z')}, {(n + 1) % N, Pauli('Z')}};
	}

	auto x_all_ham = yavque::SumLocalHam(N, yavque::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right<SumPauliStringHamEvol>(zz_even);
		circ.add_op_right<SumPauliStringHamEvol>(zz_odd);
		circ.add_op_right<SumLocalHamEvol>(x_all_ham);
	}

	auto variables = circ.variables();

	std::normal_distribution<double> ndist(0., sigma);
	for(auto& p : variables)
	{
		p = ndist(re);
	}

	Eigen::VectorXcd ini = Eigen::VectorXcd::Ones(1U << N);
	ini /= sqrt(1U << N);
	const auto ham = tfi_ham(N, 0.5);

	circ.set_input(ini);

	for(uint32_t epoch = 0; epoch < total_epochs; ++epoch)
	{
		circ.clear_evaluated();
		Eigen::VectorXcd output = *circ.output();
		for(auto& p : variables)
		{
			p.zero_grad();
		}

		circ.derivs();

		Eigen::MatrixXcd grads(1U << N, variables.size());

		for(uint32_t k = 0; k < variables.size(); ++k)
		{
			grads.col(k) = *variables[k].grad();
		}

		Eigen::MatrixXd fisher = (grads.adjoint() * grads).real();
		Eigen::RowVectorXcd o = (output.adjoint() * grads);
		fisher -= (o.adjoint() * o).real();
		fisher
			+= lambda
		       * Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(variables.size()),
		                                   static_cast<Eigen::Index>(variables.size()));

		Eigen::VectorXd egrad = (output.adjoint() * ham * grads).real();
		double energy = real(cx_double(output.adjoint() * ham * output));

		std::cout << energy << "\t" << egrad.norm() << "\t" << output.norm() << std::endl;

		Eigen::VectorXd opt = -learning_rate * fisher.inverse() * egrad;

		for(uint32_t k = 0; k < variables.size(); ++k)
		{
			variables[k] += opt(k);
		}
	}

	return 0;
}
