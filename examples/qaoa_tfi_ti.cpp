#include "example_utils.hpp"
#include "yavque.hpp"

#include "Basis/Basis1D.hpp"
#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"
#include "Hamiltonians/TITFIsing.hpp"

#include <nlohmann/json.hpp>
#include <tbb/global_control.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

/*
 * This code simulate the QAOA circuit within the translation invariant
 * subspace.
 * */

template<typename Basis> Eigen::SparseMatrix<double> ti_zz(Basis&& basis)
{
	TITFIsing<uint32_t> tfi(basis, -1.0, 0.0);
	return edp::constructSparseMat<double>(
		basis.getDim(),
		[&tfi](uint32_t n) {
			return tfi.getCol(n);
		}); // NOLINT(clang-analyzer-core.uninitialized.UndefReturn)
}

template<typename Basis> Eigen::SparseMatrix<double> ti_x_all(Basis&& basis)
{
	TITFIsing<uint32_t> tfi(basis, 0.0, -1.0);
	return edp::constructSparseMat<double>(
		basis.getDim(),
		[&tfi](uint32_t n) {
			return tfi.getCol(n);
		}); // NOLINT(clang-analyzer-core.uninitialized.UndefReturn)
}

template<typename Basis> Eigen::SparseMatrix<double> tfi_ham(double h, Basis&& basis)
{
	TITFIsing<uint32_t> tfi(basis, 1.0, h);
	return edp::constructSparseMat<double>(
		basis.getDim(),
		[&tfi](uint32_t n) {
			return tfi.getCol(n);
		}); // NOLINT(clang-analyzer-core.uninitialized.UndefReturn)
}

int main()
{
	using std::sqrt;
	using yavque::Circuit, yavque::HamEvol, yavque::cx_double;

	const uint32_t N = 12;
	const uint32_t depth = 6;
	const uint32_t total_epochs = 1000;

	const double sigma = 1.0e-3;
	const double learning_rate = 1.0e-2;
	const double h = 0.5;
	const double lambda = 1.0e-3;

	const int num_threads = get_num_threads();
	std::cerr << "Processing using " << num_threads << " threads." << std::endl;
	tbb::global_control c(tbb::global_control::max_allowed_parallelism, num_threads);

	std::random_device rd;
	std::default_random_engine re{rd()};

	edlib::Basis1D<uint32_t> basis(N, 0, false);

	Circuit circ(basis.getDim());

	yavque::Hamiltonian ham_ti_zz(
		ti_zz(basis)
			.cast<cx_double>()); // NOLINT(clang-analyzer-core.uninitialized.UndefReturn)
	yavque::Hamiltonian ham_x_all(
		ti_x_all(basis)
			.cast<cx_double>()); // NOLINT(clang-analyzer-core.uninitialized.UndefReturn)

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right<HamEvol>(ham_ti_zz);
		circ.add_op_right<HamEvol>(ham_x_all);
	}

	auto variables = circ.variables();

	std::normal_distribution<double> ndist(0., sigma);
	for(auto& p : variables)
	{
		p = ndist(re);
	}

	Eigen::VectorXcd ini(basis.getDim());
	for(uint32_t k = 0; k < basis.getDim(); ++k)
	{
		ini(k) = sqrt(double(basis.rotRpt(int(k)))) / sqrt(1U << N);
	}

	std::cout << ini.norm() << std::endl;

	const auto ham = tfi_ham(h, basis);

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

		Eigen::MatrixXcd grads(basis.getDim(), variables.size());

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
