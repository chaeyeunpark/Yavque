#include "common.hpp"

#include "yavque/Circuit.hpp"
#include "yavque/operators.hpp"

#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/EDP/LocalHamiltonian.hpp"

#include <catch2/catch_all.hpp>
#include <tbb/tbb.h>

using namespace Eigen;
using std::cos;
using std::sin;

VectorXcd eval_using_circuit(const uint32_t N, const VectorXcd& ini,
                             const std::vector<yavque::Hamiltonian>& pauli_strs,
                             const std::vector<uint32_t>& confs,
                             const std::vector<double>& ts)
{
	assert(confs.size() == ts.size());
	yavque::Circuit circuit(N);
	for(auto conf : confs)
	{
		circuit.add_op_right<yavque::HamEvol>(pauli_strs[conf]);
	}
	auto params = circuit.variables();

	circuit.set_input(ini);

	for(uint32_t tidx = 0; tidx < ts.size(); ++tidx)
	{
		params[tidx] = ts[tidx];
	}

	return *circuit.output();
}

VectorXcd eval_using_ham(VectorXcd ini,
                         const std::vector<yavque::Hamiltonian>& pauli_strs,
                         const std::vector<uint32_t>& confs,
                         const std::vector<double>& ts)
{
	assert(confs.size() == ts.size());

	constexpr yavque::cx_double I(0., 1.);

	for(uint32_t p = 0; p < confs.size(); ++p)
	{
		const double t = ts[p];
		ini = cos(t) * ini - I * sin(t) * pauli_strs[confs[p]].apply_right(ini);
	}

	return ini;
}

template<typename RandomEngine>
void test_twoqubit(const uint32_t N, RandomEngine& re,
                   const Eigen::SparseMatrix<double>& ham, bool odd)
{
	yavque::Circuit circuit1(N);
	yavque::Circuit circuit2(N);

	const uint32_t offset = [odd]()
	{
		if(odd)
		{
			return 1U;
		}
		return 0U;
	}();

	{ // construct circuit1
		edp::LocalHamiltonian<double> ham_ct(N, 2);
		for(uint32_t i = offset; i < N; i += 2)
		{
			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, (i + 1) % N}, ham);
			const auto ham = yavque::Hamiltonian(
				edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));
			circuit1.add_op_right<yavque::HamEvol>(ham);
		}
	}

	{ // construct circuit2
		edp::LocalHamiltonian<double> ham_ct(N, 2);
		for(uint32_t i = offset; i < N; i += 2)
		{
			ham_ct.addTwoSiteTerm({i, (i + 1) % N}, ham);
		}
		const auto ham = yavque::Hamiltonian(
			edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));

		circuit2.add_op_right<yavque::HamEvol>(ham);
	}

	// NOLINTNEXTLINE(misc-const-correctness)
	std::normal_distribution<double> nd{};
	for(uint32_t instance_idx = 0; instance_idx < 10; ++instance_idx)
	{
		const double t = nd(re);

		VectorXcd ini = VectorXcd::Random(1U << N);
		ini.normalize();

		for(auto& p : circuit1.variables())
		{
			p = t;
		}

		for(auto& p : circuit2.variables())
		{
			p = t;
		}

		circuit1.set_input(ini);
		circuit2.set_input(ini);

		const auto output1 = *circuit1.output();
		const auto output2 = *circuit2.output();

		REQUIRE_THAT((output1 - output2).norm(), Catch::Matchers::WithinAbs(0., 1e-6));
	}
}

TEST_CASE("Random XYZ circuit", "[random-circuit]")
{
	constexpr uint32_t N = 8; // number of qubits

	std::mt19937_64 re{1557U};
	std::normal_distribution<double> nd{};

	std::vector<yavque::Hamiltonian> hams;

	// Add xx
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0; i < N; i++)
	{
		for(uint32_t j = i + 1; j < N; j++)
		{
			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, j}, yavque::pauli_xx());
			hams.emplace_back(
				edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));

			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, j}, yavque::pauli_yy());
			hams.emplace_back(
				edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));

			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, j}, yavque::pauli_zz());
			hams.emplace_back(
				edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct));
		}
	}

	std::uniform_int_distribution<uint32_t> uid(0U, hams.size() - 1);

	for(uint32_t depth = 1; depth <= 100; depth += 20)
	{
		for(uint32_t instance_idx = 0; instance_idx < 10; ++instance_idx)
		{
			std::vector<uint32_t> confs;
			std::vector<double> ts;

			for(uint32_t p = 0; p < depth; ++p)
			{
				confs.emplace_back(uid(re));
				ts.emplace_back(nd(re));
			}

			VectorXcd ini = VectorXcd::Random(1U << N);
			ini.normalize();

			auto res1 = eval_using_circuit(N, ini, hams, confs, ts);
			auto res2 = eval_using_ham(ini, hams, confs, ts);

			REQUIRE_THAT((res1 - res2).norm(), Catch::Matchers::WithinAbs(0., 1e-6));
		}
	}
}

TEST_CASE("Test QAOA XX layers", "[qaoa-layer]")
{
	constexpr uint32_t N = 8; // number of qubits

	std::mt19937_64 re{1557U};

	SECTION("test XX even layer")
	{
		test_twoqubit(N, re, yavque::pauli_xx(), false);
	}
	SECTION("test XX odd layer")
	{
		test_twoqubit(N, re, yavque::pauli_xx(), true);
	}

	SECTION("test YY even layer")
	{
		test_twoqubit(N, re, yavque::pauli_yy(), false);
	}
	SECTION("test YY odd layer")
	{
		test_twoqubit(N, re, yavque::pauli_yy(), true);
	}

	SECTION("test ZZ even layer")
	{
		test_twoqubit(N, re, yavque::pauli_zz(), false);
	}
	SECTION("test ZZ odd layer")
	{
		test_twoqubit(N, re, yavque::pauli_zz(), true);
	}
}

TEST_CASE("Test QAOA XX+YY layers", "[qaoa-layer]")
{
	constexpr uint32_t N = 8; // number of qubits

	std::mt19937_64 re{1557U};

	const SparseMatrix<double> m = yavque::pauli_xx() + yavque::pauli_yy();
	SECTION("test XX+YY even layer")
	{
		test_twoqubit(N, re, m, false);
	}

	SECTION("test XX+YY odd layer")
	{
		test_twoqubit(N, re, m, true);
	}
}
