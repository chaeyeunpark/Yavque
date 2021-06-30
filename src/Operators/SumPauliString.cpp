#include "yavque/Operators/SumPauliString.hpp"

namespace qunn::detail
{
bool commute(const std::map<uint32_t, Pauli>& p1,
	const std::map<uint32_t, Pauli>& p2)
{
	std::vector<uint32_t> intersection;
	for(auto iter = p1.begin(); iter != p1.end(); ++iter)
	{
		uint32_t key = iter->first;
		if(p2.find(key) != p2.end())
			intersection.push_back(key);
	}

	uint32_t parity = 0;
	for(uint32_t idx: intersection)
	{
		if(p1.at(idx) != p2.at(idx))
			++parity;
	}

	return (parity % 2) == 0;
}

std::string extract_pauli_string(const std::map<uint32_t, Pauli>& pmap)
{
	std::vector<char> pstr;
	for(auto [site_idx, p]: pmap)
	{
		pstr.push_back(static_cast<char>(p));
	}
	return std::string(pstr.begin(), pstr.end());
}

} //namespace qunn::detail
