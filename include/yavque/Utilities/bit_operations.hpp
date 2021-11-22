#pragma once
#include <cstdint>
namespace yavque
{
namespace detail
{
	constexpr uint32_t bitswap(uint32_t b, uint32_t i, uint32_t j)
	{
		unsigned int x = ((b >> i) ^ (b >> j)) & 1;
		b ^= ((x << i) | (x << j));
		return b;
	}
}
} // namespace yavque::detail
