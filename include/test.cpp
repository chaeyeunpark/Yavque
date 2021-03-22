#include "Circuit.hpp"
int main()
{
	using namespace qunn;
	const int N = 8;
	Circuit C(N);
	C |= U1;

	auto CC = C.dagger()|C;
	return 0;
}
