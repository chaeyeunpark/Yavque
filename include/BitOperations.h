#pragma once
inline void bitswap(unsigned int& b, int i, int j)
{
	unsigned int x = ((b >> i) ^ (b >> j)) & 1;
	b ^= ((x << i) | (x << j));
}
