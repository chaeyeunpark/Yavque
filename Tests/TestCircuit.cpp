#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <tbb/tbb.h>

#include "common.hpp"
#include "yavque/Circuit.hpp"
#include "yavque/operators.hpp"

#include "EDP/ConstructSparseMat.hpp"
#include "EDP/LocalHamiltonian.hpp"

using namespace Eigen;
using std::cos;
using std::sin;

using Catch::Matchers::Floating::WithinAbsMatcher;

tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 2);

VectorXcd eval_using_circuit(const uint32_t N, const VectorXcd& ini,
                             const std::vector<yavque::Hamiltonian>& pauli_strs,
                             const std::vector<uint32_t>& confs,
                             const std::vector<double>& ts)
{
	assert(confs.size() == ts.size());
	yavque::Circuit circuit(N);
	for(auto conf : confs)
	{
		circuit.add_op_right(std::make_unique<yavque::HamEvol>(pauli_strs[conf]));
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
		double t = ts[p];
		ini = cos(t) * ini - I * sin(t) * pauli_strs[confs[p]].apply_right(ini);
	}

	return ini;
}

template<typename RandomEngine>
void test_twoqubit(const uint32_t N, RandomEngine& re,
                   const Eigen::SparseMatrix<double>& ham, bool odd)
{
	std::normal_distribution<> nd;
	yavque::Circuit circuit1(N);
	yavque::Circuit circuit2(N);

	uint32_t offset = 0u;

	if(odd)
		offset = 1u;

	{ // construct circuit1
		edp::LocalHamiltonian<double> ham_ct(N, 2);
		for(uint32_t i = offset; i < N; i += 2)
		{
			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, (i + 1) % N}, ham);
			auto ham = yavque::Hamiltonian(
				edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct));
			circuit1.add_op_right(std::make_unique<yavque::HamEvol>(ham));
		}
	}

	{ // construct circuit2
		std::vector<std::shared_ptr<yavque::Hamiltonian>> hams1;

		edp::LocalHamiltonian<double> ham_ct(N, 2);
		for(uint32_t i = offset; i < N; i += 2)
		{
			ham_ct.addTwoSiteTerm({i, (i + 1) % N}, ham);
		}
		auto ham = yavque::Hamiltonian(
			edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct));

		circuit2.add_op_right(std::make_unique<yavque::HamEvol>(ham));
	}

	for(uint32_t instance_idx = 0; instance_idx < 10; ++instance_idx)
	{
		double t = nd(re);

		VectorXcd ini = VectorXcd::Random(1 << N);
		ini.normalize();

		for(auto& p : circuit1.variables())
			p = t;

		for(auto& p : circuit2.variables())
			p = t;

		circuit1.set_input(ini);
		circuit2.set_input(ini);

		auto output1 = *circuit1.output();
		auto output2 = *circuit2.output();

		REQUIRE_THAT((output1 - output2).norm(), WithinAbsMatcher(0., 1e-6));
	}
}

TEST_CASE("Random XYZ circuit", "[random-circuit]")
{
	constexpr uint32_t N = 8; // number of qubits

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;

	std::vector<yavque::Hamiltonian> hams;

	// Add xx
	edp::LocalHamiltonian<double> ham_ct(N, 2);
	for(uint32_t i = 0; i < N; i++)
	{
		for(uint32_t j = i + 1; j < N; j++)
		{
			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, j}, yavque::pauli_xx());
			hams.emplace_back(yavque::Hamiltonian(
				edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct)));

			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, j}, yavque::pauli_yy());
			hams.emplace_back(yavque::Hamiltonian(
				edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct)));

			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm({i, j}, yavque::pauli_zz());
			hams.emplace_back(yavque::Hamiltonian(
				edp::constructSparseMat<yavque::cx_double>(1 << N, ham_ct)));
		}
	}

	std::uniform_int_distribution<uint32_t> uid(0u, hams.size() - 1);

	for(uint32_t depth = 1; depth < 100; ++depth)
	{
		for(uint32_t instance_idx = 0; instance_idx < 100; ++instance_idx)
		{
			std::vector<uint32_t> confs;
			std::vector<double> ts;

			for(uint32_t p = 0; p < depth; ++p)
			{
				confs.emplace_back(uid(re));
				ts.emplace_back(nd(re));
			}

			VectorXcd ini = VectorXcd::Random(1 << N);
			ini.normalize();

			auto res1 = eval_using_circuit(N, ini, hams, confs, ts);
			auto res2 = eval_using_ham(ini, hams, confs, ts);

			REQUIRE_THAT((res1 - res2).norm(), WithinAbsMatcher(0., 1e-6));
		}
	}
}

TEST_CASE("Test QAOA XX layers", "[qaoa-layer]")
{
	constexpr uint32_t N = 8; // number of qubits

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;

	SECTION("test XX even layer") { test_twoqubit(N, re, yavque::pauli_xx(), false); }
	SECTION("test XX odd layer") { test_twoqubit(N, re, yavque::pauli_xx(), true); }

	SECTION("test YY even layer") { test_twoqubit(N, re, yavque::pauli_yy(), false); }
	SECTION("test YY odd layer") { test_twoqubit(N, re, yavque::pauli_yy(), true); }

	SECTION("test ZZ even layer") { test_twoqubit(N, re, yavque::pauli_zz(), false); }
	SECTION("test ZZ odd layer") { test_twoqubit(N, re, yavque::pauli_zz(), true); }
}

TEST_CASE("Test QAOA XX+YY layers", "[qaoa-layer]")
{
	constexpr uint32_t N = 8; // number of qubits

	std::random_device rd;
	std::default_random_engine re{rd()};
	std::normal_distribution<> nd;

	SparseMatrix<double> m = yavque::pauli_xx() + yavque::pauli_yy();
	SECTION("test XX+YY even layer") { test_twoqubit(N, re, m, false); }

	SECTION("test XX+YY odd layer") { test_twoqubit(N, re, m, true); }
}
