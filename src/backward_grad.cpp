#include "yavque/backward_grad.hpp"

std::pair<double, Eigen::VectorXd>
yavque::value_and_grad(const Eigen::SparseMatrix<yavque::cx_double>& op, 
		const Circuit& circuit)
{
	/*
	 * We use the notation in arXiv:2009.02823
	 * */
	Eigen::VectorXcd psi = *circuit.output();

	double value = real(yavque::cx_double(psi.adjoint()*op*psi));
	Circuit lambda_circuit = circuit.dagger();

	lambda_circuit.set_input(op*psi);

	std::vector<double> derivs;
	auto op_size = circuit.num_operators();

	for(size_t idx = 0; idx < op_size; ++idx)
	{
		if(auto* diff_op = dynamic_cast<Univariate*>
				(circuit.operator_at(idx).get()))
		{
			Eigen::VectorXcd right = *circuit.state_at(idx);
			right = diff_op->log_deriv()->apply_right(right);

			Eigen::VectorXcd left = *lambda_circuit.state_at(op_size - idx);
			double deriv = 2.0*real(yavque::cx_double(left.adjoint() * right));
			derivs.emplace_back(deriv);
		}
	}
	return std::make_pair(value, Eigen::VectorXd{
			Eigen::Map<Eigen::VectorXd>(derivs.data(), derivs.size())
		});
}
