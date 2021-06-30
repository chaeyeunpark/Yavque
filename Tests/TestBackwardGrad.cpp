#include "Operators/Hamiltonian.hpp"
#include <sstream>
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <tbb/tbb.h>

#include <Circuit.hpp>
#include <operators.hpp>
#include <EDP/LocalHamiltonian.hpp>

tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 2);

Eigen::SparseMatrix<qunn::cx_double> hadamard()
{
	using std::sqrt;
	Eigen::SparseMatrix<qunn::cx_double> h(2,2);
	h.coeffRef(0, 0) = 1.0/sqrt(2.0);
	h.coeffRef(0, 1) = 1.0/sqrt(2.0);
	h.coeffRef(1, 0) = 1.0/sqrt(2.0);
	h.coeffRef(1, 1) = -1.0/sqrt(2.0);

	h.makeCompressed();
	return h;
};

std::unique_ptr<qunn::Hamiltonian> hadamard_at(uint32_t n_qubits, uint32_t idx)
{
	edp::LocalHamiltonian<qunn::cx_double> ham_ct(n_qubits, 2);
	ham_ct.addOneSiteTerm(idx, hadamard());
	auto ham = edp::constructSparseMat<qunn::cx_double>(1 << n_qubits, ham_ct);

	std::ostringstream os;
	os << "Hadamard at " << idx << std::endl;

	return std::make_unique<qunn::Hamiltonian>(ham, os.str());
}

std::unique_ptr<qunn::Hamiltonian> 
cnot_at(uint32_t n_qubits, uint32_t control, uint32_t target)
{
	const uint32_t dim = (1u << n_qubits);
	Eigen::SparseMatrix<qunn::cx_double> op(dim, dim);
	for(uint32_t k = 0; k < dim; ++k)
	{
		uint32_t t = k ^ (((dim >> control) & 1) << target);
		op.coeffRef(t, k) = 1;
	}
	op.makeCompressed();

	std::ostringstream os;
	os << "CNOT [" << control << ", " << target << "]" << std::endl;


	return std::make_unique<qunn::Hamiltonian>(op, os.str());
}


TEST_CASE("Test gradients using the QAOA circuit")
{
	using namespace qunn;
	constexpr uint32_t N = 16;
	constexpr uint32_t dim = 1 << N; //dimension of the total Hilbert space
	Circuit circuit(dim);
	
	// set initial state |0\rangle^{\otimes N}
	Eigen::VectorXd ini = Eigen::VectorXd::Zero(dim);
	ini(0) = 1.0;

	
}
