#pragma once
#include <functional>
#include <utility>

#include "Optimizer.hpp"

#include "AdaMax.hpp"
#include "Adam.hpp"
#include "SGD.hpp"
#include "SGDMomentum.hpp"


namespace yavque 
{

class OptimizerFactory final
{
private:
	std::unordered_map<std::string, 
		std::function<std::unique_ptr<Optimizer>(const nlohmann::json&)> > optCstr_;

	template<class OptimizerT>
	void resiterOptimizer(const std::string& name)
	{
		optCstr_[name] = [](const nlohmann::json& param) -> std::unique_ptr<Optimizer>
		{
			return std::make_unique<OptimizerT>(param); 
		};
	}

	explicit OptimizerFactory()
	{
		resiterOptimizer<SGD>("SGD");
		resiterOptimizer<SGDMomentum>("SGDMomentum");
		resiterOptimizer<Adam>("Adam");
		resiterOptimizer<AdaMax>("AdaMax");
	}

public:
	OptimizerFactory(const OptimizerFactory& ) = delete;
	OptimizerFactory& operator=(const OptimizerFactory&) = delete;

	OptimizerFactory(OptimizerFactory&& ) = delete;
	OptimizerFactory& operator=(OptimizerFactory&& ) = delete;

	static OptimizerFactory& getInstance()
	{
		static OptimizerFactory instance;
		return instance;
	}

	std::unique_ptr<Optimizer> createOptimizer(const nlohmann::json& opt) const
	{
		auto iter = optCstr_.find(opt["name"]);
		if (iter == optCstr_.end())
		{
			throw std::invalid_argument("Such an optimizer does not exist.");
		}
		return iter->second(opt);
	}

	~OptimizerFactory() = default;
};
} //namespace yavque
