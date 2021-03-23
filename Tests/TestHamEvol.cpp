#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <random>
#include <Eigen/Dense>

#include "EDP/LocalHamiltonian.hpp"
#include "EDP/ConstructSparseMat.hpp"

#include "Operators/operators.hpp"

#include "common.hpp"


TEST_CASE("Test single qubit Hamiltonian", "[one-site]") {
	using namespace Eigen;
	using std::sqrt;
	using std::cos;
	using std::sin;
	using std::exp;
	constexpr uint32_t N = 8;//number of qubits

	//ini is |+>^N
	VectorXcd ini = VectorXcd::Ones(1<<N);
	ini /= sqrt(1<<N);

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;

	constexpr qunn::cx_double I(0.,1.);

	{
		edp::LocalHamiltonian<double> ham_ct(N, 2);
		for(uint32_t i = 0; i < N; i++)
		{
			ham_ct.addOneSiteTerm(i, qunn::pauli_x());
		}

		auto ham = qunn::Hamiltonian(
				edp::constructSparseMat<qunn::cx_double>(1<<N, ham_ct));
		auto hamEvol = qunn::HamEvol(ham);
		
		auto var = hamEvol.parameter();

		for(int i = 0; i < 100; i++)
		{
			double t = nd(re);
			var = t;
			VectorXcd out_test = hamEvol*ini;

			VectorXcd p = (cos(t)*MatrixXcd::Identity(2,2) 
					- I*sin(t)*qunn::pauli_x())*VectorXcd::Ones(2);
			p /= sqrt(2);

			VectorXcd out = product_state(N,p);

			REQUIRE((out - out_test).norm() < 1e-6);
		}
	}
	{
		edp::LocalHamiltonian<double> ham_ct(N, 2);
		for(uint32_t i = 0; i < N; i++)
		{
			ham_ct.addOneSiteTerm(i, qunn::pauli_z());
		}

		auto ham = qunn::Hamiltonian(edp::constructSparseMat<qunn::cx_double>(1<<N, ham_ct));
		auto hamEvol = qunn::HamEvol(ham);
		auto var = hamEvol.parameter();

		for(int i = 0; i < 100; i++)
		{
			double t = nd(re);
			var = t;
			VectorXcd out_test = hamEvol*ini;

			VectorXcd p = (cos(t)*MatrixXcd::Identity(2,2) 
					- I*sin(t)*qunn::pauli_z())*VectorXcd::Ones(2);
			p /= sqrt(2);

			VectorXcd out = product_state(N,p);

			REQUIRE((out - out_test).norm() < 1e-6);
		}
	}

}
TEST_CASE("Test basic operations", "[basic]") {
	using namespace Eigen;
	using namespace qunn;
	using std::sqrt;
	using std::cos;
	using std::sin;
	using std::exp;
	constexpr uint32_t N = 8;//number of qubits

	//ini is |+>^N
	VectorXcd ini = VectorXcd::Ones(1<<N);
	ini /= sqrt(1<<N);

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;

	constexpr qunn::cx_double I(0.,1.);

	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0; i < N; i++)
	{
		ham_ct.addOneSiteTerm(i, qunn::pauli_x());
	}

	auto ham = qunn::Hamiltonian(
			edp::constructSparseMat<qunn::cx_double>(1<<N, ham_ct));
	auto hamEvol = HamEvol(ham);

	auto copied = hamEvol.clone();

	REQUIRE(hamEvol.parameter() != dynamic_cast<HamEvol*>(copied.get())->parameter());

	REQUIRE(hamEvol.hamiltonian().is_same_ham( dynamic_cast<HamEvol*>(copied.get())->hamiltonian()));

}
