#pragma once
#include <cstdint>


namespace yavque
{
class Counter
{
private:
	uint32_t counter_ = 0;

public:
	Counter()
	{
	}

	static Counter& getInstance() 
	{
		static Counter inst;
		return inst;
	}

	uint32_t count()
	{
		//add thread safety
		return counter_++;
	}
};
} // namespace yavque
