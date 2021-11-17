#include <cmath>
#include <iostream>
#include <random>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "yavque.hpp"

Eigen::SparseMatrix<double> tfi_ham(const uint32_t N, double h)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t k = 0; k < N; ++k)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(k, (k+1) % N), yavque::pauli_zz());
		ham_ct.addOneSiteTerm(k, h*yavque::pauli_x());
	}
	return -edp::constructSparseMat<double>(1U << N, ham_ct);
}

int main()
{
	using std::sqrt;

	using yavque::Circuit;
	using yavque::cx_double;

	const uint32_t N = 12;
	const uint32_t depth = 6;
	const double sigma = 1.0e-3;
	const double learning_rate = 1.0e-2;
	const uint32_t total_epochs = 1000;
	const double lambda = 1.0e-3;

	std::random_device rd;
	std::default_random_engine re{rd()};

	Circuit circ(1U << N);

	Eigen::VectorXd zz_even(1U << N);
	Eigen::VectorXd zz_odd(1U << N);

	for(uint32_t n = 0; n < (1U << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; k += 2)
		{
			int z0 = 1 - 2*int((n >> k) & 1U);
			int z1 = 1 - 2*int((n >> ((k+1)%N)) & 1U);
			elt += z0*z1;
		}
		zz_even(n) = elt;
	}

	for(uint32_t n = 0; n < (1U << N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 1; k < N; k += 2)
		{
			int z0 = 1 - 2*int((n >> k) & 1U);
			int z1 = 1 - 2*int((n >> ((k+1) % N)) & 1U);
			elt += z0*z1;
		}
		zz_odd(n) = elt;
	}

	auto zz_even_ham = yavque::DiagonalOperator(zz_even, "zz even");
	auto zz_odd_ham = yavque::DiagonalOperator(zz_odd, "zz odd");
	auto x_all_ham = yavque::SumLocalHam(N, yavque::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<yavque::DiagonalHamEvol>(zz_even_ham));
		circ.add_op_right(std::make_unique<yavque::DiagonalHamEvol>(zz_odd_ham));
		circ.add_op_right(std::make_unique<yavque::SumLocalHamEvol>(x_all_ham));
	}

	auto variables = circ.variables();

	std::normal_distribution<double> ndist(0., sigma);
	for(auto& p: variables)
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
		for(auto& p: variables)
		{
			p.zero_grad();
		}

		circ.derivs();

		Eigen::MatrixXcd grads(1U << N, variables.size());

		for(uint32_t k = 0; k < variables.size(); ++k)
		{
			grads.col(k) = *variables[k].grad();
		}

		Eigen::MatrixXd fisher = (grads.adjoint()*grads).real();
		Eigen::RowVectorXcd o = (output.adjoint()*grads);
		fisher -= (o.adjoint()*o).real();
		fisher += lambda*Eigen::MatrixXd::Identity(
				static_cast<Eigen::Index>(variables.size()),
				static_cast<Eigen::Index>(variables.size()));

		Eigen::VectorXd egrad = (output.adjoint()*ham*grads).real();
		double energy = real(cx_double(output.adjoint()*ham*output));

		std::cout << energy << "\t" << egrad.norm() << "\t" << output.norm() << std::endl;

		Eigen::VectorXd opt = -learning_rate*fisher.inverse()*egrad;

		for(uint32_t k = 0; k < variables.size(); ++k)
		{
			variables[k] += opt(k);
		}
	}

	return 0;
}
