cmake_minimum_required(VERSION 3.9)
project(yavque_tests)

include_directories("${CMAKE_SOURCE_DIR}/externals")

add_executable(TestHamEvol "TestHamEvol.cpp")
target_link_libraries(TestHamEvol yavque Eigen3::Eigen)
add_test(
	NAME TestHamEvol
	COMMAND TestHamEvol
)

add_executable(TestDiagonalHamEvol "TestDiagonalHamEvol.cpp")
target_link_libraries(TestDiagonalHamEvol yavque Eigen3::Eigen)
add_test(
	NAME TestDiagonalHamEvol
	COMMAND TestDiagonalHamEvol
)

add_executable(TestSumLocalHamEvol "TestSumLocalHamEvol.cpp")
target_link_libraries(TestSumLocalHamEvol yavque Eigen3::Eigen)
add_test(
	NAME TestSumLocalHamEvol
	COMMAND TestSumLocalHamEvol
)

add_executable(TestSumPauliString "TestSumPauliString.cpp")
target_link_libraries(TestSumPauliString yavque Eigen3::Eigen)
add_test(
	NAME TestSumPauliString
	COMMAND TestSumPauliString
)

add_executable(TestSumPauliStringHamEvol "TestSumPauliStringHamEvol.cpp")
target_link_libraries(TestSumPauliStringHamEvol yavque Eigen3::Eigen)
add_test(
	NAME TestSumPauliStringHamEvol
	COMMAND TestSumPauliStringHamEvol
)

add_executable(TestCircuit "TestCircuit.cpp")
target_link_libraries(TestCircuit yavque Eigen3::Eigen)
add_test(
	NAME TestCircuit
	COMMAND TestCircuit
)

add_executable(TestGrad "TestGrad.cpp")
target_link_libraries(TestGrad yavque Eigen3::Eigen)
add_test(
	NAME TestGrad
	COMMAND TestGrad
)

add_executable(TestSingleQubitOperator "TestSingleQubitOperator.cpp")
target_link_libraries(TestSingleQubitOperator yavque Eigen3::Eigen)
add_test(
	NAME TestSingleQubitOperator 
	COMMAND TestSingleQubitOperator
)

add_executable(TestTwoQubitOperator "TestTwoQubitOperator.cpp")
target_link_libraries(TestTwoQubitOperator yavque Eigen3::Eigen)
add_test(
	NAME TestTwoQubitOperator 
	COMMAND TestTwoQubitOperator
)

add_executable(TestTFI "TestTFI.cpp")
target_link_libraries(TestTFI yavque Eigen3::Eigen)
add_test(
	NAME TestTFI
	COMMAND TestTFI
)

add_executable(TestBackwardGrad "TestBackwardGrad.cpp")
target_link_libraries(TestBackwardGrad yavque Eigen3::Eigen)
add_test(
	NAME TestBackwardGrad
	COMMAND TestBackwardGrad
)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
	DEPENDS TestHamEvol TestDiagonalHamEvol TestSumPauliString TestSumPauliStringHamEvol
	TestCircuit TestGrad TestSingleQubitOperator TestTwoQubitOperator TestTFI
	TestBackwardGrad)


