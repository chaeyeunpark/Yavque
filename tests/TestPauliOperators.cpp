#include "yavque/Utilities/pauli_operators.hpp"
#include "common.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include <catch2/catch_all.hpp>

template <typename DerivedA, typename DerivedB, typename S = remove_complex_t<typename DerivedA::Scalar>>
inline bool
isApproxEqual(const Eigen::DenseBase<DerivedA>& a, const Eigen::DenseBase<DerivedB>& b,
              S eps =
                  std::numeric_limits<S>::epsilon() *
                  100) {
	  return ((a.derived() - b.derived()).array().abs()
          <= eps).all();
}

TEST_CASE("Test Pauli operators", "[utils]") {
	Eigen::MatrixXcd pauli_x_dense {
		{0, 1},
		{1, 0}
	};
	Eigen::MatrixXcd pauli_y_dense {
		{0, -std::complex<double>(0.0, 1.0)},
		{std::complex<double>(0.0, 1.0), 0}
	};
	Eigen::MatrixXcd pauli_z_dense {
		{1, 0},
		{0, -1}
	};
	SECTION("Test PauliX") {
		REQUIRE(isApproxEqual(pauli_x_dense, Eigen::MatrixXcd(yavque::pauli_x())));
	}
	SECTION("Test PauliY") {
		REQUIRE(isApproxEqual(pauli_y_dense, Eigen::MatrixXcd(yavque::pauli_y())));
	}
	SECTION("Test PauliZ") {
		REQUIRE(isApproxEqual(pauli_z_dense, Eigen::MatrixXcd(yavque::pauli_z())));
	}

	Eigen::MatrixXcd pauli_xx_dense = Eigen::kroneckerProduct(pauli_x_dense, pauli_x_dense);
	Eigen::MatrixXcd pauli_yy_dense = Eigen::kroneckerProduct(pauli_y_dense, pauli_y_dense);
	Eigen::MatrixXcd pauli_zz_dense = Eigen::kroneckerProduct(pauli_z_dense, pauli_z_dense);

	SECTION("Test PauliXX") {
		REQUIRE(isApproxEqual(pauli_xx_dense, Eigen::MatrixXcd(yavque::pauli_xx())));
	}
	SECTION("Test PauliYY") {
		REQUIRE(isApproxEqual(pauli_yy_dense, Eigen::MatrixXcd(yavque::pauli_yy())));
	}
	SECTION("Test PauliZZ") {
		REQUIRE(isApproxEqual(pauli_zz_dense, Eigen::MatrixXcd(yavque::pauli_zz())));
	}
	SECTION("Test PauliXXYY") {
		REQUIRE(isApproxEqual(pauli_xx_dense+pauli_yy_dense, Eigen::MatrixXcd(yavque::pauli_xx_yy())));
	}
	SECTION("Test PauliXXYYZZ") {
		REQUIRE(isApproxEqual(pauli_xx_dense+pauli_yy_dense+pauli_zz_dense, Eigen::MatrixXcd(yavque::pauli_xx_yy_zz())));
	}
}