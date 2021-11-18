#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <sstream>

#include <tbb/tbb.h>

#include "yavque/operators.hpp"
#include "yavque/Circuit.hpp"
#include "yavque/Variable.hpp"

#include "common.hpp"

#include "EDP/LocalHamiltonian.hpp"
#include "EDP/ConstructSparseMat.hpp"

tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);

template<typename RandomEngine>
void test_commuting(const uint32_t N, const uint32_t depth, 
		Eigen::SparseMatrix<double> op, RandomEngine& re)
{
	using namespace yavque;
	constexpr std::complex<double> I(0., 1.);

	std::uniform_real_distribution<> urd(-M_PI, M_PI);
	std::vector<int> indices;

	for(uint32_t n = 0; n < N; n++)
	{
		indices.push_back(n);
	}

	for(int k = 0; k < 10; ++k) // instance iteration
	{
		Circuit circ{N};

		edp::LocalHamiltonian<double> ham_ct(N, 2);

		std::vector<yavque::Hamiltonian> hams;
		for(uint32_t i = 0; i < depth; i ++)
		{
			ham_ct.clearTerms();
			std::shuffle(indices.begin(), indices.end(), re);
			auto connection = std::pair<int,int>{indices[0], indices[1]};
			ham_ct.addTwoSiteTerm(connection, op);
			auto ham = yavque::Hamiltonian(edp::constructSparseMat<yavque::cx_double>(1<<N, ham_ct));

			hams.push_back(ham);

			circ.add_op_right(std::make_unique<yavque::HamEvol>(ham));
		}

		Eigen::VectorXcd v(1<<N);
		v.setRandom();
		v.normalize();
		circ.set_input(v);
		auto variables = circ.variables();

		for(auto p: variables)
		{
			p = urd(re);
		}
		auto circ_output = *circ.output();

		circ.derivs();

		for(uint32_t n = 0; n < depth; ++n)
		{
			Eigen::VectorXcd der1 = hams[n].apply_right(circ_output);
			der1 *= -I;

			Eigen::VectorXcd der2 = *variables[n].grad();

			REQUIRE((der1 - der2).norm() < 1e-6);
		}
	}
}

TEST_CASE("test gradient of cummuting circuit", "[commuting]") {
	using namespace yavque;
	const int N = 8;
	const int depth = 20;

	std::random_device rd;
	std::default_random_engine re{rd()};

	SECTION("test xx commuting circuit") {
		test_commuting(N, depth, pauli_xx(), re);
	}// end section
	SECTION("test yy commuting circuit") {
		test_commuting(N, depth, pauli_yy(), re);
	}// end section
	SECTION("test zz commuting circuit") {
		test_commuting(N, depth, pauli_zz(), re);
	}// end section
}


TEST_CASE("test grad for two-qubit paulis", "[two-qubit-pauli]") {
	using namespace yavque;
	const uint32_t N = 8;
	const uint32_t depth = 20;

	std::random_device rd;
	std::default_random_engine re{0};

	std::uniform_int_distribution<> ham_gen(0, 2);
	std::uniform_real_distribution<> urd(-M_PI, M_PI);

	std::vector<std::string> pauli_names = {"xx", "yy", "zz"};


	for(int k = 0; k < 10; ++k) // instance iteration
	{
		Circuit circ{N};
		edp::LocalHamiltonian<double> ham_ct(N, 2);

		std::vector<Eigen::SparseMatrix<double> > hams = {pauli_xx(), pauli_yy(), pauli_zz()};

		for(uint32_t i = 0; i < depth; i ++)
		{
			auto connection = random_connection(N, re);
			auto p = ham_gen(re);
			ham_ct.clearTerms();
			ham_ct.addTwoSiteTerm(connection, hams[p]);
			
			std::ostringstream ss;
			ss << pauli_names[p] << " between " << connection.first << " and " 
				<< connection.second;

			auto ham = yavque::Hamiltonian(
					edp::constructSparseMat<yavque::cx_double>(1<<N, ham_ct), ss.str());

			circ.add_op_right(std::make_unique<yavque::HamEvol>(ham));
		}

		Eigen::VectorXcd v(1<<N);
		v.setRandom();
		v.normalize();

		circ.set_input(v);
		auto variables = circ.variables();

		for(auto p: variables)
		{
			p = urd(re);
		}

		auto circ_output = *circ.output();

		circ.derivs();

		for(uint32_t n = 0; n < depth; ++n)
		{
			Eigen::VectorXcd der1 = *variables[n].grad();
			auto circ_der = circ;

			auto val = circ_der.variables()[n].value();
			circ_der.variables()[n] = val + M_PI/2;
			circ_der.clear_evaluated();
			circ_der.evaluate();
			Eigen::VectorXcd out1 = *circ_der.output();
			circ_der.variables()[n] = val - M_PI/2;
			circ_der.clear_evaluated();
			circ_der.evaluate();
			Eigen::VectorXcd out2 = *circ_der.output();

			Eigen::VectorXcd der2 = (out1 - out2)/2;

			REQUIRE((der1 - der2).norm() < 1e-6);
		}
	}
}

std::pair<yavque::Circuit, std::vector<yavque::Variable>>
qaoa_shared_var(const uint32_t N, const uint32_t depth)
{
	using namespace yavque;
	Circuit circ{N};
	std::vector<Variable> variables(3*depth);

	std::vector<yavque::Hamiltonian> ham_zzs;

	for(uint32_t i = 0; i < N; ++i)
	{
		edp::LocalHamiltonian<double> ham_ct(N, 2);
		ham_ct.addTwoSiteTerm(std::make_pair(i, (i+1)%N), pauli_zz());
		
		std::ostringstream ss;
		ss << "zz between " << i << " and " << (i+1)%N;

		ham_zzs.emplace_back(
			edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct),
			ss.str());
	}

	std::vector<yavque::Hamiltonian> ham_xs;

	for(uint32_t i = 0; i < N; ++i)
	{
		edp::LocalHamiltonian<double> ham_ct(N, 2);
		ham_ct.addOneSiteTerm(i, pauli_x());
		
		std::ostringstream ss;
		ss << "x on " << i;

		ham_xs.emplace_back(
			edp::constructSparseMat<yavque::cx_double>(1U << N, ham_ct),
			ss.str());
	}


	for(uint32_t p = 0; p < depth; ++p)
	{
		for(uint32_t i = 0; i < N; i += 2) //add zz even
			circ.add_op_right(std::make_unique<yavque::HamEvol>(ham_zzs[i], variables[3*p]));

		for(uint32_t i = 1; i < N; i += 2) //add zz odd
			circ.add_op_right(std::make_unique<yavque::HamEvol>(ham_zzs[i], variables[3*p+1]));

		for(uint32_t i = 0; i < N; ++i)
			circ.add_op_right(std::make_unique<yavque::HamEvol>(ham_xs[i], variables[3*p+2]));
	}

	return std::make_pair(std::move(circ), variables);
}

yavque::Circuit qaoa_sum_ham(const uint32_t N, const uint32_t depth)
{
	using namespace yavque;
	Circuit circ{N};

	edp::LocalHamiltonian<double> ham_ct(N, 2);
	ham_ct.clearTerms();
	for(uint32_t i = 0; i < N; i += 2)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(i, (i+1)%N), pauli_zz());
	}
	std::ostringstream ss;
	ss << "zz even";
	auto ham_zz_even = yavque::Hamiltonian(
		edp::constructSparseMat<yavque::cx_double>(1<<N, ham_ct),
		ss.str());

	ham_ct.clearTerms();
	for(uint32_t i = 1; i < N; i += 2)
	{
		ham_ct.addTwoSiteTerm(std::make_pair(i, (i+1)%N), pauli_zz());
	}
	ss.clear();
	ss << "zz odd";
	auto ham_zz_odd = yavque::Hamiltonian(
			edp::constructSparseMat<yavque::cx_double>(1<<N, ham_ct),
			ss.str());

	ham_ct.clearTerms();
	for(uint32_t i = 0; i < N; ++i)
	{
		ham_ct.addOneSiteTerm(i, pauli_x());
	}
	ss.clear();
	ss << "x all";

	auto ham_x_all = yavque::Hamiltonian(
		edp::constructSparseMat<yavque::cx_double>(1<<N, ham_ct),
		ss.str());


	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<yavque::HamEvol>(ham_zz_even));
		circ.add_op_right(std::make_unique<yavque::HamEvol>(ham_zz_odd));
		circ.add_op_right(std::make_unique<yavque::HamEvol>(ham_x_all));
	}
	
	return circ;
}

yavque::Circuit qaoa_diag_prod_ham(const uint32_t N, const uint32_t depth)
{
	using namespace yavque;
	Circuit circ{1u<<N};

	Eigen::VectorXd zz_even(1<<N);
	Eigen::VectorXd zz_odd(1<<N);
	
	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 0; k < N; k += 2)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+1)%N)) & 1);
			elt += z0*z1;
		}
		zz_even(n) = elt;
	}
	
	for(uint32_t n = 0; n < (1u<<N); ++n)
	{
		int elt = 0;
		for(uint32_t k = 1; k < N; k += 2)
		{
			int z0 = 1-2*((n >> k) & 1);
			int z1 = 1-2*((n >> ((k+1)%N)) & 1);
			elt += z0*z1;
		}
		zz_odd(n) = elt;
	}

	auto zz_even_ham = yavque::DiagonalOperator(zz_even, "zz even");
	auto zz_odd_ham = yavque::DiagonalOperator(zz_odd, "zz odd");
	auto x_all_ham = yavque::SumLocalHam(N, yavque::pauli_x().cast<cx_double>());

	for(uint32_t p = 0; p < depth; ++p)
	{
		circ.add_op_right(std::make_unique<yavque::DiagonalHamEvol>(zz_even_ham));
		circ.add_op_right(std::make_unique<yavque::DiagonalHamEvol>(zz_odd_ham));
		circ.add_op_right(std::make_unique<yavque::SumLocalHamEvol>(x_all_ham));
	}
	
	return circ;
}

TEST_CASE("test grad for qaoa for TFI", "[qaoa]") {
	using namespace yavque;
	const int N = 8;
	const int depth = 10;

	std::random_device rd;
	std::default_random_engine re{0};

	std::uniform_real_distribution<> urd(-M_PI, M_PI);

	for(int k = 0; k < 10; ++k) // instance iteration
	{
		std::vector<double> param_values(3*depth);
		for(uint32_t k = 0; k < 3*depth; ++k)
		{
			param_values[k] = urd(re);
		}

		auto [circ1, variables1] = qaoa_shared_var(N, depth);
		auto circ2 = qaoa_sum_ham(N, depth);
		auto circ3 = qaoa_diag_prod_ham(N, depth);

		Eigen::VectorXcd v(1<<N);
		v.setRandom();
		v.normalize();

		circ1.set_input(v);
		circ2.set_input(v);
		circ3.set_input(v);

		for(uint32_t k = 0; k < 3*depth; ++k)
		{
			variables1[k] = param_values[k];
			circ2.variables()[k] = param_values[k];
			circ3.variables()[k] = param_values[k];
		}

		circ1.evaluate();
		circ1.derivs();

		circ2.evaluate();
		circ2.derivs();

		circ3.evaluate();
		circ3.derivs();

		for(uint32_t k = 0; k < 3*depth; ++k)
		{
			auto grad1 = *variables1[k].grad();
			auto grad2 = *circ2.variables()[k].grad();
			auto grad3 = *circ3.variables()[k].grad();

			REQUIRE((grad1 - grad2).norm() < 1e-6);
			REQUIRE((grad2 - grad3).norm() < 1e-6);
		}

	}
}

