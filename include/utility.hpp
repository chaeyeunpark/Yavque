#pragma once
#include <complex>
#include <memory>
namespace qunn
{
	using cx_double = std::complex<double>;

	Eigen::VectorXcd apply_local(const Eigen::VectorXcd& vec,
		const Eigen::MatrixXcd& unitary,
		const std::vector<uint32_t>& sites);


}
