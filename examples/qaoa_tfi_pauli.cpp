#include <random>
#include <iostream>
#include <cmath>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "yavque.hpp"
#include "yavque/Operators/SumPauliString.hpp"

Eigen::SparseMatrix<double> tfi_ham(const uint32_t N, double h)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t k = 0; k < N; ++k)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(k, (k+1) % N), yavque::pauli_zz());
		ham_ct.addOneSiteTerm(k, h*yavque::pauli_x());
	}
	return -edp::constructSparseMat<double>(1 << N, ham_ct);
}

int main()
{
	using namespace yavque;
	using std::sqrt;
	const uint32_t N = 12;
	const uint32_t depth = 6;
	const double sigma = 1.0e-3;
	const double learning_rate = 1.0e-2;

	std::random_device rd;
	std::default_random_engine re{rd()};

	Circuit circ(1 << N);


	yavque::SumPauliString zz_even(N);
	for(uint32_t n = 0; n < N; n += 2)
	{
		zz_even += {{n,  Pauli('Z')}, {n+1, Pauli('Z')}};
	}
	yavque::SumPauliString zz_odd(N);
	for(uint32_t n = 1; n < N; n += 2)
	{
		zz_odd += {{n,  Pauli('Z')}, {(n+1)%N, Pauli('Z')}};
	}

	auto x_all_ham = yavque::SumLocalHam(N, yavque::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<SumPauliStringHamEvol>(zz_even));
		circ.add_op_right(std::make_unique<SumPauliStringHamEvol>(zz_odd));
		circ.add_op_right(std::make_unique<yavque::SumLocalHamEvol>(x_all_ham));
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
		Eigen::RowVectorXcd o = (output.adjoint()*grads);
		fisher -= (o.adjoint()*o).real();
		fisher += 1e-3*Eigen::MatrixXd::Identity(parameters.size(), parameters.size());

		Eigen::VectorXd egrad = (output.adjoint()*ham*grads).real();
		double energy = real(cx_double(output.adjoint()*ham*output));

		std::cout << energy << "\t" << egrad.norm() << "\t" << output.norm() << std::endl;

		Eigen::VectorXd opt = -learning_rate*fisher.inverse()*egrad;

		for(uint32_t k = 0; k < parameters.size(); ++k)
		{
			parameters[k] += opt(k);
		}
	}

	return 0;
}
