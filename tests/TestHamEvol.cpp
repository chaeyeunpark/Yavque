#include "common.hpp"

#include "yavque/Operators/HamEvol.hpp"
#include "yavque/utils.hpp"

#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/EDP/LocalHamiltonian.hpp"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

#include <random>

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
	// NOLINTNEXTLINE(misc-const-correctness)
	std::normal_distribution<double> nd{};

	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0; i < N; i++)
	{
		ham_ct.addOneSiteTerm(i, m);
	}

	auto ham
		= yavque::Hamiltonian(edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));
	auto hamEvol = yavque::HamEvol(ham);
	auto var = hamEvol.get_variable();

	for(int i = 0; i < 100; i++)
	{
		VectorXcd ini = VectorXcd::Random(1U << N);
		ini.normalize();
		const double t = nd(re);
		var = t;
		const VectorXcd out_test = hamEvol * ini;

		const MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) - I * sin(t) * m);
		const VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}

	hamEvol.dagger_in_place();
	for(int i = 0; i < 100; i++)
	{
		VectorXcd ini = VectorXcd::Random(1U << N);
		ini.normalize();

		const double t = nd(re);
		var = t;
		const VectorXcd out_test = hamEvol * ini;

		const MatrixXcd mevol = (cos(t) * MatrixXcd::Identity(2, 2) + I * sin(t) * m);
		const VectorXcd out = apply_kronecker(N, mevol, ini);

		REQUIRE((out - out_test).norm() < 1e-6);
	}
}

TEST_CASE("test single qubit Hamiltonian", "[one-site]")
{
	constexpr uint32_t N = 8; // number of qubits

	// ini is |+>^N
	std::mt19937_64 re{1557U};

	SECTION("test using pauli X")
	{
		test_single_qubit(N, yavque::pauli_x(), re);
	}

	SECTION("test using pauli Z")
	{
		test_single_qubit(N, yavque::pauli_z(), re);
	}
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
	VectorXcd ini = VectorXcd::Ones(1U << N);
	ini /= sqrt(1U << N);

	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0; i < N; i++)
	{
		ham_ct.addOneSiteTerm(i, yavque::pauli_x());
	}

	auto ham
		= yavque::Hamiltonian(edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));
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

	std::mt19937_64 re{1557U};

	std::vector<Eigen::SparseMatrix<double>> pauli_ops
		= {yavque::pauli_xx(), yavque::pauli_yy(), yavque::pauli_zz()};

	// NOLINTBEGIN(misc-const-correctness)
	std::uniform_int_distribution<int> uid(0, 2);
	std::normal_distribution<double> nd;
	// NOLINTEND(misc-const-correctness)

	for(uint32_t k = 0; k < 100; ++k) // instances
	{
		// ini is random
		VectorXcd ini = VectorXcd::Random(1U << N);
		ini.normalize();

		edp::LocalHamiltonian<double> ham_ct(N, 2);
		ham_ct.addTwoSiteTerm(random_connection(N, re), pauli_ops[uid(re)]);

		auto ham = yavque::Hamiltonian(
			edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));
		auto hamEvol = HamEvol(ham);

		const double val = nd(re);

		hamEvol.set_variable_value(val);
		const Eigen::VectorXcd grad1
			= hamEvol.log_deriv()->apply_right(hamEvol.apply_right(ini));

		hamEvol.get_variable() += M_PI / 2;
		Eigen::VectorXcd grad2 = hamEvol.apply_right(ini);
		hamEvol.get_variable() = val - M_PI / 2;
		grad2 -= hamEvol.apply_right(ini);
		grad2 /= 2.0;

		REQUIRE((grad1 - grad2).norm() < 1e-6);
	}
}
