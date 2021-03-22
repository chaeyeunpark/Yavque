#include <vector>
#include <Eigen/Dense>

#include "BitOperations.h"

namespace qunn
{
Eigen::VectorXcd apply_local(const Eigen::VectorXcd& vec,
		const Eigen::MatrixXcd& unitary,
		const std::vector<uint32_t>& sites)
{
	Eigen::VectorXcd res = Eigen::VectorXcd::Zero(vec.size());
	for(int i = 0; i < (vec.size() >> sites.size()); i++)
	{
		//parallelize!
		for(int j = 0; j < (1<<sites.size()); j++)
		{
			unsigned int index = (i << sites.size()) | j;
			for(uint32_t site_idx = 0; site_idx < sites.size(); ++site_idx)
			{
				bitswap(index, site_idx, sites[site_idx]);
			}
			auto c = unitary.col(j);
			for(int k = 0; k < 4; k++)
			{
				unsigned int index_to = (i << 2) | k;
				for(uint32_t site_idx = 0; site_idx < sites.size(); ++site_idx)
				{
					bitswap(index_to, site_idx, sites[site_idx]);
				}
				res(index_to) += c(k)*vec(index);
			}
		}
	}
	return res;
}
} //namespace qunn
