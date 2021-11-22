#define CATCH_CONFIG_MAIN
#include <Eigen/Dense>
#include <catch.hpp>
#include <random>

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

#include "yavque/Operators/HamEvol.hpp"
#include "yavque/utils.hpp"

#include "common.hpp"

template<typename RandomEngine>
void test_single_qubit(const uint32_t N, const Eigen::SparseMatrix<double>& m,
                       RandomEngine& re)
{
	using namespace Eigen;
	using std::cos;
	using std::exp;
	using std::sin;
	using std::sqrt;

	constexpr yavque::cx_double I(0., 1.);
	std::normal_distribution<> nd;

	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0; i < N; i++)
	{
		ham_ct.addOneSiteTerm(i, m);
	}

	auto ham
		= yavque::Hamiltonian(edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct));
	auto hamEvol = yavque::HamEvol(ham);
	auto var = hamEvol.get_variable();

	for(int i = 0; i < 100; i++)
	{
		VectorXcd ini = VectorXcd::Random(1 << N);
		ini.normalize();
		double t = nd(re);
		var = t;
		VectorXcd out_test = hamEvol * ini;

		MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) - I * sin(t) * m);
		VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}

	hamEvol.dagger_in_place();
	for(int i = 0; i < 100; i++)
	{
		VectorXcd ini = VectorXcd::Random(1 << N);
		ini.normalize();

		double t = nd(re);
		var = t;
		VectorXcd out_test = hamEvol * ini;

		MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) + I * sin(t) * m);
		VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}
}

TEST_CASE("test single qubit Hamiltonian", "[one-site]")
{
	constexpr uint32_t N = 8; // number of qubits

	// ini is |+>^N
	std::random_device rd;
	std::default_random_engine re{rd()};

	SECTION("test using pauli X") { test_single_qubit(N, yavque::pauli_x(), re); }

	SECTION("test using pauli Z") { test_single_qubit(N, yavque::pauli_z(), re); }
}

TEST_CASE("Test basic operations", "[basic]")
{
	using namespace Eigen;
	using namespace yavque;
	using std::cos;
	using std::exp;
	using std::sin;
	using std::sqrt;
	constexpr uint32_t N = 8; // number of qubits

	// ini is |+>^N
	VectorXcd ini = VectorXcd::Ones(1 << N);
	ini /= sqrt(1 << N);

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;

	constexpr yavque::cx_double I(0., 1.);

	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0; i < N; i++)
	{
		ham_ct.addOneSiteTerm(i, yavque::pauli_x());
	}

	auto ham
		= yavque::Hamiltonian(edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct));
	auto hamEvol = HamEvol(ham);

	auto copied = hamEvol.clone();

	REQUIRE(hamEvol.get_variable()
	        != dynamic_cast<HamEvol*>(copied.get())->get_variable());

	REQUIRE(hamEvol.hamiltonian().is_same_ham(
		dynamic_cast<HamEvol*>(copied.get())->hamiltonian()));
}

TEST_CASE("Test gradient", "[log-deriv]")
{
	using namespace Eigen;
	using namespace yavque;
	using std::cos;
	using std::exp;
	using std::sin;
	using std::sqrt;
	constexpr uint32_t N = 8; // number of qubits

	std::random_device rd;
	std::default_random_engine re{rd()};

	std::vector<Eigen::SparseMatrix<double>> pauli_ops
		= {yavque::pauli_xx(), yavque::pauli_yy(), yavque::pauli_zz()};

	std::uniform_int_distribution<> uid(0, 2);
	std::normal_distribution<> nd;

	for(uint32_t k = 0; k < 100; ++k) // instances
	{
		// ini is random
		VectorXcd ini = VectorXcd::Random(1 << N);
		ini.normalize();

		edp::LocalHamiltonian<double> ham_ct(N, 2);
		ham_ct.addTwoSiteTerm(random_connection(N, re), pauli_ops[uid(re)]);

		auto ham = yavque::Hamiltonian(
			edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct));
		auto hamEvol = HamEvol(ham);

		double val = nd(re);

		hamEvol.set_variable_value(val);
		Eigen::VectorXcd grad1
			= hamEvol.log_deriv()->apply_right(hamEvol.apply_right(ini));

		hamEvol.get_variable() += M_PI / 2;
		Eigen::VectorXcd grad2 = hamEvol.apply_right(ini);
		hamEvol.get_variable() = val - M_PI / 2;
		grad2 -= hamEvol.apply_right(ini);
		grad2 /= 2.0;

		REQUIRE((grad1 - grad2).norm() < 1e-6);
	}
}
