#include "example_utils.hpp"

#include <tbb/tbb.h>

#include <cstdlib>

int get_num_threads()
{
	const char* p = getenv("TBB_NUM_THREADS"); // NOLINT(concurrency-mt-unsafe)
	if(p == nullptr)
	{
		return tbb::this_task_arena::max_concurrency();
	}
	int num_threads = parse_int<int>(p);
	return num_threads;
}
