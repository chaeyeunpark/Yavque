#include "common.hpp"

#include "yavque/Operators/SumPauliString.hpp"
#include "yavque/utils.hpp"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

#include <random>

TEST_CASE("test commuting operators", "[test-commuting]")
{
	using namespace yavque;
	{
		std::map<uint32_t, Pauli> pstr1;
		std::map<uint32_t, Pauli> pstr2;

		pstr1[0] = Pauli('X');
		pstr1[1] = Pauli('X');

		pstr2[0] = Pauli('X');
		pstr2[1] = Pauli('X');

		REQUIRE(detail::commute(pstr1, pstr2));
	}
	{
		std::map<uint32_t, Pauli> pstr1;
		std::map<uint32_t, Pauli> pstr2;

		pstr1[0] = Pauli('X');
		pstr1[1] = Pauli('X');

		pstr2[0] = Pauli('Z');
		pstr2[1] = Pauli('Z');

		REQUIRE(detail::commute(pstr1, pstr2));
	}
	{
		std::map<uint32_t, Pauli> pstr1;
		std::map<uint32_t, Pauli> pstr2;

		pstr1[0] = Pauli('Z');
		pstr1[1] = Pauli('X');
		pstr1[2] = Pauli('Z');

		pstr2[1] = Pauli('Z');
		pstr2[2] = Pauli('X');
		pstr2[3] = Pauli('Z');

		REQUIRE(detail::commute(pstr1, pstr2));
	}
	{
		std::map<uint32_t, Pauli> pstr1;
		std::map<uint32_t, Pauli> pstr2;

		pstr1[0] = Pauli('Z');
		pstr1[1] = Pauli('X');
		pstr1[2] = Pauli('Z');

		pstr2[1] = Pauli('Z');
		pstr2[2] = Pauli('X');
		pstr2[3] = Pauli('Z');

		REQUIRE(detail::commute(pstr1, pstr2));
	}
}

TEST_CASE("test CompressedPauliString", "[pauli-string]")
{
	using namespace yavque;

	std::mt19937_64 re{1557U};

	// NOLINTNEXTLINE(misc-const-correctness)
	std::uniform_int_distribution<uint32_t> uid(0, 2);
	const uint32_t N = 12;

	std::vector<uint32_t> sites;
	sites.reserve(N);

	const std::array<char, 3> pauli_names = {'X', 'Y', 'Z'};

	for(uint32_t k = 0; k < N; ++k)
	{
		sites.push_back(k);
	}

	for(uint32_t str_len = 2; str_len < 5; ++str_len)
	{
		for(uint32_t k = 0; k < 100; ++k) // instance
		{
			std::shuffle(sites.begin(), sites.end(), re);
			std::vector<uint32_t> indices(sites.begin(), sites.begin() + str_len);
			std::string pstr;
			for(uint32_t m = 0; m < str_len; ++m)
			{
				pstr.push_back(pauli_names[uid(re)]);
			}

			Eigen::VectorXcd input = Eigen::VectorXcd::Random(1U << N);
			input.normalize();

			Eigen::VectorXcd res = input;

			auto cps = detail::CompressedPauliString(pstr);

			const Eigen::VectorXcd res1
				= cps.apply(indices, res); // apply pstr[0] to 0 and pstr[1] to 1
			for(uint32_t k = 0; k < str_len; ++k)
			{
				switch(pstr[k])
				{
				case 'X':
					res = apply_single_qubit(res, Eigen::MatrixXd(pauli_x()), indices[k]);
					break;
				case 'Y':
					res = apply_single_qubit(res, Eigen::MatrixXcd(pauli_y()),
					                         indices[k]);
					break;
				case 'Z':
					res = apply_single_qubit(res, Eigen::MatrixXd(pauli_z()), indices[k]);
					break;
				default:
					// this should not be reached
					__builtin_unreachable();
					break;
				}
			}

			REQUIRE((res - res1).norm() < 1e-6);
		}
	}
}
TEST_CASE("test SumPauliString", "[sum-pauli-string]")
{
	using namespace yavque;

	std::mt19937_64 re{1557U};

	const uint32_t N = 10;
	const uint32_t n_terms = 5;

	// NOLINTBEGIN(misc-const-correctness)
	std::uniform_int_distribution<uint32_t> uid(0, 2);
	std::uniform_int_distribution<uint32_t> string_len_dist(2, N - 1);
	// NOLINTEND(misc-const-correctness)

	std::vector<uint32_t> sites;
	sites.reserve(N);

	auto gen = [&uid, &re]()
	{
		return Pauli('X' + uid(re));
	};

	for(uint32_t k = 0; k < N; ++k)
	{
		sites.push_back(k);
	}

	for(uint32_t instance = 0; instance < 100; ++instance)
	{
		SumPauliString sps(N);

		std::vector<std::vector<uint32_t>> indices_v;
		std::vector<std::vector<Pauli>> pstring_v;

		for(uint32_t k = 0; k < n_terms; ++k)
		{
			const uint32_t string_len = string_len_dist(re);
			std::shuffle(begin(sites), end(sites), re);
			std::vector<uint32_t> indices(sites.begin(), sites.begin() + string_len);

			std::vector<Pauli> pstring(string_len);
			std::ranges::generate(pstring, gen);

			indices_v.push_back(indices);
			pstring_v.push_back(pstring);

			std::map<uint32_t, Pauli> pstr;
			std::ranges::transform(indices, pstring, std::inserter(pstr, pstr.end()),
			                       [](const auto& aa, const auto& bb)
			                       { return std::make_pair(aa, bb); });

			sps += pstr;
		}

		Eigen::VectorXcd input = Eigen::VectorXcd::Random(1U << N);
		input.normalize();

		const Eigen::VectorXcd res1 = sps.apply_right(input);
		Eigen::VectorXcd res2 = Eigen::VectorXcd::Zero(1U << N);

		for(uint32_t k = 0; k < n_terms; ++k)
		{
			auto cps = detail::CompressedPauliString(pstring_v[k]);
			res2 += cps.apply(indices_v[k], input);
		}

		REQUIRE((res1 - res2).norm() < 1e-6);
	}
}
