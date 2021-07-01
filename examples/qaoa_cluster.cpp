#include <random>
#include <iostream>
#include <cmath>

#include <fstream>
#include <tbb/tbb.h>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "yavque.hpp"


Eigen::SparseMatrix<yavque::cx_double> single_pauli(const uint32_t N, const uint32_t idx, 
		const Eigen::SparseMatrix<yavque::cx_double>& m)
{
	edp::LocalHamiltonian<yavque::cx_double> lh(N, 2);
	lh.addOneSiteTerm(idx, m);
	return edp::constructSparseMat<yavque::cx_double>(1<<N, lh);
}

Eigen::SparseMatrix<yavque::cx_double> identity(const uint32_t N)
{
	std::vector<Eigen::Triplet<yavque::cx_double>> triplets;
	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		triplets.emplace_back(n, n, 1.0);
	}
	Eigen::SparseMatrix<yavque::cx_double> m(1<<N,1<<N);
	m.setFromTriplets(triplets.begin(), triplets.end());
	return m;
}


Eigen::SparseMatrix<yavque::cx_double> cluster_ham(uint32_t N, double h)
{
	using namespace yavque;
	Eigen::SparseMatrix<cx_double> ham(1<<N, 1<<N);
	for(uint32_t k = 0; k < N; k++)
	{
		Eigen::SparseMatrix<cx_double> term = identity(N);
		term = term*single_pauli(N, k, pauli_z().cast<cx_double>());
		term = term*single_pauli(N, (k+1)%N, pauli_x().cast<cx_double>());
		term = term*single_pauli(N, (k+2)%N, pauli_z().cast<cx_double>());

		ham += -term;
	}

	edp::LocalHamiltonian<double> lh(N, 2);
	for(uint32_t k = 0; k < N; k++)
	{
		lh.addOneSiteTerm(k, pauli_x());
	}
	ham -= h*edp::constructSparseMat<cx_double>(1<<N, lh);

	return ham;
}

int get_num_threads()
{
	const char* p = getenv("TBB_NUM_THREADS");
	if(!p)
		return tbb::this_task_arena::max_concurrency();
	return atoi(p);
}

int main(int argc, char *argv[])
{
    using namespace yavque;
    using std::sqrt;

	const uint32_t total_epochs = 10;
	const double h = 0.5;

	const uint32_t N = 12;
	const uint32_t depth = 6;
	const double sigma = 1.0e-2;
	const double learning_rate = 1.0e-2;

	const int num_threads = get_num_threads();
	std::cerr << "Processing using " << num_threads << " threads." << std::endl;
	tbb::global_control c(tbb::global_control::max_allowed_parallelism, 
			num_threads);

	std::random_device rd;
	std::default_random_engine re{rd()};

	Circuit circ(1 << N);

	std::vector<std::map<uint32_t, Pauli>> ti_zxz;

	for(uint32_t k = 0; k < N; ++k)
	{
		std::map<uint32_t, Pauli> term;
		term[k] = Pauli('Z');
		term[(k+1)%N] = Pauli('X');
		term[(k+2)%N] = Pauli('Z');

		ti_zxz.emplace_back(std::move(term));
	}

	auto zxz_ham = yavque::SumPauliString(N, ti_zxz);
	auto x_all_ham = yavque::SumLocalHam(N, yavque::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<yavque::SumPauliStringHamEvol>(zxz_ham));
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
	const auto ham = cluster_ham(N, h);

	std::cout.precision(10);

	circ.set_input(ini);

	for(uint32_t epoch = 0; epoch < total_epochs; ++epoch)
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
		double lambda = std::max(100.0*std::pow(0.9, epoch), 1e-3);
		fisher += lambda*Eigen::MatrixXd::Identity(parameters.size(), parameters.size());

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
