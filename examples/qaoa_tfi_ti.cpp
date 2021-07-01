#include <random>
#include <iostream>
#include <cmath>

#include <stdlib.h>
#include <tbb/global_control.h>

#include <nlohmann/json.hpp>

#include "Basis/TIBasis.hpp"
#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "Hamiltonians/TITFIsing.hpp"
#include "yavque.hpp"

/*
 * This code simulate the QAOA circuit within the translation invariant
 * subspace.
 * */

template<typename Basis>
Eigen::SparseMatrix<double> ti_zz(Basis&& basis)
{
	TITFIsing<uint32_t> tfi(basis, -1.0, 0.0);
    return edp::constructSparseMat<double>(basis.getDim(), [&tfi](uint32_t n){ return tfi.getCol(n); });
}

template<typename Basis>
Eigen::SparseMatrix<double> ti_x_all(Basis&& basis)
{
	TITFIsing<uint32_t> tfi(basis, 0.0, -1.0);
    return edp::constructSparseMat<double>(basis.getDim(), [&tfi](uint32_t n){ return tfi.getCol(n); });
}

template<typename Basis>
Eigen::SparseMatrix<double> tfi_ham(double h, Basis&& basis)
{
	TITFIsing<uint32_t> tfi(basis, 1.0, h);
    return edp::constructSparseMat<double>(basis.getDim(), [&tfi](uint32_t n){ return tfi.getCol(n); });
}

int get_num_threads()
{
	const char* p = getenv("TBB_NUM_THREADS");
	if(!p)
		return tbb::this_task_arena::max_concurrency();
	return atoi(p);
}

int main()
{
	using namespace yavque;
	using std::sqrt;
	const uint32_t N = 12;
	const uint32_t depth = 6;
	const double sigma = 1.0e-3;
	const double learning_rate = 1.0e-2;
	const double h = 0.5;

	const int num_threads = get_num_threads();
	std::cerr << "Processing using " << num_threads << " threads." << std::endl;
	tbb::global_control c(tbb::global_control::max_allowed_parallelism, 
			num_threads);

	std::random_device rd;
	std::default_random_engine re{rd()};

	TIBasis<uint32_t> basis(N, 0, false);

	Circuit circ(basis.getDim());

	yavque::Hamiltonian ham_ti_zz(ti_zz(basis).cast<cx_double>());
	yavque::Hamiltonian ham_x_all(ti_x_all(basis).cast<cx_double>());


	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<HamEvol>(ham_ti_zz));
		circ.add_op_right(std::make_unique<HamEvol>(ham_x_all));
	}

	auto parameters = circ.parameters();

	std::normal_distribution<double> ndist(0., sigma);
	for(auto& p: parameters)
	{
		p = ndist(re);
	}

	Eigen::VectorXcd ini(basis.getDim());
	for(uint32_t k = 0; k < basis.getDim(); ++k)
	{
		ini(k) = sqrt(basis.rotRpt(k))/sqrt(1<<N);
	}

	std::cout << ini.norm() << std::endl;

	const auto ham = tfi_ham(h, basis);

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

		Eigen::MatrixXcd grads(basis.getDim(), parameters.size());

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