#include <Circuit.hpp>
#include <random>
#include <iostream>

#include "operators.hpp"
#include "HamEvol.hpp"

#include "EDP/LocalHamiltonian.hpp"
#include "EDP/ConstructSparseMat.hpp"


std::shared_ptr<qunn::Hamiltonian> two_qubit_pauli_even(const uint32_t N, 
		const Eigen::SparseMatrix<double>& m)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0u; i < N; i += 2)
	{
		ham_ct.addTwoSiteTerm({i, (i+1)%N}, m);
	}
	return std::make_shared<qunn::Hamiltonian>(
				N, edp::constructSparseMat<qunn::cx_double>(1<<N, ham_ct));
}
std::shared_ptr<qunn::Hamiltonian> two_qubit_pauli_odd(const uint32_t N, 
		const Eigen::SparseMatrix<double>& m)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 1u; i < N; i += 2)
	{
		ham_ct.addTwoSiteTerm({i, (i+1)%N}, m);
	}
	return std::make_shared<qunn::Hamiltonian>(
				N, edp::constructSparseMat<qunn::cx_double>(1<<N, ham_ct));
}

std::shared_ptr<qunn::Hamiltonian> z_disorder(const uint32_t N, 
		const std::vector<double>& hs)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0u; i < N; ++i)
	{
		ham_ct.addOneSiteTerm(i, hs[i]*pauli_z());
	}
	return std::make_shared<qunn::Hamiltonian>(
				N, edp::constructSparseMat<qunn::cx_double>(1<<N, ham_ct));
}
Eigen::VectorXd neel_ghz(const int N)
{
	using std::acos;
	using std::cos;
	using std::sin;

	Eigen::VectorXd ini = Eigen::VectorXd::Zero(1<<N);
	{
		uint32_t n = 0;
		for(uint32_t k = 0; k < N/2; ++k)
			n |= (0b10 << (2*k));
		ini(n) = 1.0/sqrt(2); 
		ini(n >> 1) = 1.0/sqrt(2);	
	}
	return ini;
}
Eigen::VectorXd neel_ghz(const int N, double v)
{
	using std::acos;
	using std::cos;
	using std::sin;
	using std::sqrt;

	Eigen::MatrixXd m(2, 2);

	double phi = acos(v)/2.0;

	m(0,0) = cos(phi);
	m(1,0) = sin(phi);

	m(0,1) = -sin(phi);
	m(1,1) = cos(phi);

	Eigen::VectorXd v0(1);
	Eigen::VectorXd v1(1);

	v0(0) = 1.0;
	v1(0) = 1.0;

	for(int i = 0; i < N/2; ++i)
	{
		v0 = kroneckerProduct(v0, m.col(0)).eval();
		v0 = kroneckerProduct(v0, m.col(1)).eval();
		v1 = kroneckerProduct(v1, m.col(1)).eval();
		v1 = kroneckerProduct(v1, m.col(0)).eval();
	}

	return (v0 + v1)/sqrt(2);
}

Eigen::SparseMatrix<double> staggered_mag(const int N)
{
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0u; i < N; i += 2)
	{
		ham_ct.addOneSiteTerm(i, std::pow(-1, i)*pauli_z());
	}
	return edp::constructSparseMat<double>(1<<N, ham_ct);
}


Eigen::VectorXcd applyU1(const int N, Eigen::VectorXcd v, double phi)
{
	constexpr qunn::cx_double I(0., 1.);
	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int32_t k = N-2*__builtin_popcountl(n);
		v(n) = exp(-I*double(k)*phi);
	}
	return v;
}


int main(int argc, char* argv[])
{
	const uint32_t N = 8;
	const double delta_t = 0.1;
	const uint32_t depth = 100;
	const double v = (double)2/3;

	std::random_device rd;
	std::default_random_engine re;

	std::uniform_real_distribution<> urd(-0.5, 0.5);

	/*
	std::vector<double> hs;

	for(uint32_t hidx = 0; hidx < N; ++hidx)
		hs.emplace_back(urd(re));
	*/

	qunn::Circuit circuit(N);

	std::vector<std::shared_ptr<qunn::Hamiltonian>> hams;
	{
		hams.emplace_back(two_qubit_pauli_even(N, pauli_xx()));
		hams.emplace_back(two_qubit_pauli_odd(N, pauli_xx()));
		hams.emplace_back(two_qubit_pauli_even(N, pauli_yy()));
		hams.emplace_back(two_qubit_pauli_odd(N, pauli_yy()));
		hams.emplace_back(two_qubit_pauli_even(N, pauli_zz()));
		hams.emplace_back(two_qubit_pauli_odd(N, pauli_zz()));
		//hams.emplace_back(z_disorder(N, hs));
	}

	for(uint32_t p = 0; p < depth; p++)
	{
		for(auto& ham: hams)
		{
			circuit.add_op_right(std::make_shared<qunn::FullHamEvol>(N, ham));
		}
	}

	Eigen::VectorXcd ini = neel_ghz(N, v);

	circuit.set_input(ini);

	for(auto& p: circuit.parameters())
		p = delta_t;

	const auto m = staggered_mag(N);

	for(uint32_t n = 0; n <= depth; ++n)
	{
		Eigen::VectorXcd state = *circuit.state_at(n*6);
		//std::cout << state.norm() << std::endl;
		double obs = std::real(qunn::cx_double((state.adjoint()*m*m*state)));
		printf("%f\t%f\n", n*delta_t, obs);
	}
	
	return 0;
}
