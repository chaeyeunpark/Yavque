#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <iostream>
#include <filesystem>

enum class Pauli : char {I = 'I', X = 'X', Y = 'Y', Z = 'Z'};

std::optional<Pauli> charToPauli(char c)
{
	switch(c)
	{
	case 'I':
		return Pauli::I;
	case 'X':
		return Pauli::X;
	case 'Y':
		return Pauli::Y;
	case 'Z':
		return Pauli::Z;
	}
	return {};
}

class PauliString
{
private:
	std::vector<std::pair<uint32_t, Pauli> > string_;

public:
	PauliString() { }

	PauliString(std::string str)
	{
		if(str.empty())
			return ;
		while(true)
		{
			std::size_t pos = str.find(' ');
			size_t where;
			std::string token = str.substr(0, pos);
			where = atoi(token.c_str()+1);
			auto p = charToPauli(token[0]);

			if(!p.has_value())
				throw std::invalid_argument("Pauli operator cannot be parsed");
			string_.emplace_back(where, p.value());
			if(pos != std::string::npos)
				str.erase(0, pos+1);
			else
				break;
		}
	}

	uint32_t countX() const
	{
		uint32_t res = 0;
		for(const auto& p: string_)
		{
			if(p.second == Pauli::X)
			{
				++res;
			}
		}
		return res;
	}

	uint32_t countY() const
	{
		uint32_t res = 0;
		for(const auto& p: string_)
		{
			if(p.second == Pauli::Y)
			{
				++res;
			}
		}
		return res;
	}

	uint32_t countZ() const
	{
		uint32_t res = 0;
		for(const auto& p: string_)
		{
			if(p.second == Pauli::Z)
			{
				++res;
			}
		}
		return res;
	}

	/**
	 * Check whether the term is diagonal in the computational basis
	 */
	bool isDiagonal() const
	{
		for(const auto& p: string_)
		{
			if((p.second != Pauli::Z) && (p.second != Pauli::I))
			{
				return false;
			}
		}
		return true;
	}

	bool hasU1() const
	{
		return (countX() % 2 == 0) && (countY() % 2 == 0);
	}

	void add(uint32_t pos, Pauli p)
	{
		string_.emplace_back(pos, p);
	}

	std::pair<std::complex<double>, std::vector<int> > 
		operator()(const Eigen::VectorXi& sigma) const
	{
		constexpr std::complex<int> I (0, 1);
		std::complex<int> r = 1;
		std::vector<int> toFlip;
		for(const auto p: string_)
		{
			switch(p.second)
			{
			case Pauli::I :
				break;
			case Pauli::X :
				toFlip.push_back(p.first);
				break;
			case Pauli::Y : 
				toFlip.push_back(p.first);
				r *= sigma(p.first)*I;
				break;
			case Pauli::Z :
				r *= sigma(p.first);
				break;
			}
		}
		return std::make_pair(std::complex<double>(real(r), imag(r)), toFlip);
	}
	std::pair<std::complex<double>, uint32_t> 
		operator()(uint32_t col) const
	{
		constexpr std::complex<int> I (0, 1);
		std::complex<int> r = 1;
		std::vector<int> toFlip;
		for(const auto p: string_)
		{
			int sgn = 1-2*((col >> p.first) & 1);
			switch(p.second)
			{
			case Pauli::I :
				break;
			case Pauli::X :
				col ^= (1 << p.first);
				break;
			case Pauli::Y : 
				col ^= (1 << p.first);
				r *= sgn*I;
				break;
			case Pauli::Z :
				r *= sgn;
				break;
			}
		}
		return std::make_pair(std::complex<double>(real(r), imag(r)), col);
	}

	friend std::ostream& operator<<(std::ostream& os, const PauliString& ps)
	{
		const char* sep = "";
		for(const auto& p: ps.string_)
		{
			os << sep << static_cast<char>(p.second) << p.first;
			sep = " ";
		}
		return os;
	}
};


class PauliHamiltonian
{
private:
	uint32_t n_;
	uint32_t nup_; //only effective when the Hamiltonain has U1 symmetry.
	std::vector<std::pair<double, PauliString> > terms_;

public:
	PauliHamiltonian(uint32_t n, uint32_t jz = std::numeric_limits<uint32_t>::max())
		: n_{n}, nup_{jz}
	{
	}

	uint32_t getN() const
	{
		return n_;
	}

	PauliHamiltonian(const PauliHamiltonian&) = default;
	PauliHamiltonian(PauliHamiltonian&&) = default;

	PauliHamiltonian& operator=(const PauliHamiltonian&) = default;
	PauliHamiltonian& operator=(PauliHamiltonian&&) = default;

	nlohmann::json params() const
	{
		using nlohmann::json;
		json res;
		res["name"] = "PauliHamiltonian";
		res["n"] = n_;
		json terms = json::array();;
		for(auto [coeff, pauliString]: terms_)
		{
			std::ostringstream ss;
			ss << pauliString;
			std::string s = ss.str();
			terms.push_back({coeff, std::move(s)});
		}
		res["terms"] = terms;
		return res;
	}

	template<typename ...Ts>
	void emplaceTerm(Ts&&... args)
	{
		terms_.emplace_back(std::forward<Ts>(args)...);
	}

	bool hasTermwiseU1() const
	{
		return std::all_of(terms_.begin(), terms_.end(), 
				[](const std::pair<double, PauliString>& p){
					return p.second.hasU1();
				});
	}

	bool isReal() const
	{
		return std::all_of(terms_.begin(), terms_.end(),
				[](const std::pair<double, PauliString>& p){
					return (p.second.countY() % 2) == 0;
				});
	}

	uint32_t getNup() const
	{
		return nup_;
	}

	static PauliHamiltonian fromFile(const std::filesystem::path& filePath)
	{
		std::ifstream fin(filePath);
		std::string line;
		std::getline(fin, line);
		uint32_t n;
		uint32_t m;
		std::istringstream ss(line);
		ss >> n >> m;

		PauliHamiltonian res(n, m);
		while(std::getline(fin, line))
		{
			auto to = line.find("\t");
			double coeff = stof(line.substr(0, to));
			PauliString s(line.substr(to+1, std::string::npos));
			res.emplaceTerm(coeff, s);
		}
		return res;
	}

	template<typename State>
	typename State::Scalar operator()(const State& state) const
	{
		typename State::Scalar res = 0;
		for(auto [coeff, pauliString]: terms_)
		{
			const auto& [c, toFlip] = pauliString(state.getSigma());
			res += c*coeff*state.ratio(toFlip);
		}
		return res;
	}

	std::map<uint32_t, std::complex<double> > operator()(const uint32_t col) const
	{
		std::map<uint32_t, std::complex<double> > m;
		for(auto [coeff, pauliString]: terms_)
		{
			const auto& [c, s] = pauliString(col);
			m[s] += c*coeff;
		}
		return m;
	}


	friend PauliHamiltonian diagonalHamiltonian(const PauliHamiltonian& ph)
	{
		PauliHamiltonian res(ph.n_, ph.nup_);
		for(auto [coeff, pauliString]: ph.terms_)
		{
			if(pauliString.isDiagonal())
				res.emplaceTerm(coeff, pauliString);
		}
		return res;
	}
};
