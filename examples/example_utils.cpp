#include "example_utils.hpp"

#include <tbb/tbb.h>

#include <cstdlib>

int get_num_threads()
{
	const char* p = getenv("TBB_NUM_THREADS"); //NOLINT(concurrency-mt-unsafe)
	if(p == nullptr)
	{
		return tbb::this_task_arena::max_concurrency();
	}
	char* end = nullptr;

	int num_threads = static_cast<int>(strtol(p, &end, 10));
	return num_threads;
}
