#pragma once
#include <charconv>
#include <cstring>

template<typename T = int>
auto parse_int(const char* cstr) -> T
{
	static_assert(std::is_integral_v<T>, "Type T must be an integer.");
	T val = 0;
	std::from_chars(cstr, cstr + strlen(cstr), val);
	return val;
}

int get_num_threads();
