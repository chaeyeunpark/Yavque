cmake_minimum_required(VERSION 3.9)

project(QUNN)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories("${CMAKE_SOURCE_DIR}/externals") 
include_directories("${CMAKE_SOURCE_DIR}/externals/ExactDiagonalization/Include/")
include_directories("${CMAKE_SOURCE_DIR}/externals/nlohmann_json_cmake_fetchcontent/include/")
include_directories("${CMAKE_SOURCE_DIR}/externals/spectra/include/")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DEIGEN_USE_MKL_ALL")

set(CATCH_URL "https://github.com/catchorg/Catch2/releases/download/v2.11.1/catch.hpp")
set(CATCH_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/externals/catch.hpp")


if (NOT EXISTS "${CATCH_DOWNLOAD_PATH}")
	file(DOWNLOAD "${CATCH_URL}" "${CATCH_DOWNLOAD_PATH}")
endif()


## Setup MKL
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	include_directories("$ENV{MKLROOT}/include")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
	link_libraries(mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -tbb")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(tbb stdc++ pthread m dl)
endif()


add_library(qunn STATIC "src/VariableImpl.cpp" "src/Circuit.cpp" "src/Operators/utils.cpp")
target_link_libraries(qunn Eigen3::Eigen)


######################## Test objects ##########################################
add_executable(TestHamEvol "Tests/TestHamEvol.cpp")
target_link_libraries(TestHamEvol qunn Eigen3::Eigen)
add_test(
	NAME TestHamEvol
	COMMAND TestHamEvol
)

add_executable(TestDiagonalHamEvol "Tests/TestDiagonalHamEvol.cpp")
target_link_libraries(TestDiagonalHamEvol qunn Eigen3::Eigen)
add_test(
	NAME TestDiagonalHamEvol
	COMMAND TestDiagonalHamEvol
)

add_executable(TestProductHamEvol "Tests/TestProductHamEvol.cpp")
target_link_libraries(TestProductHamEvol qunn Eigen3::Eigen)
add_test(
	NAME TestProductHamEvol
	COMMAND TestProductHamEvol
)

add_executable(TestCircuit "Tests/TestCircuit.cpp")
target_link_libraries(TestCircuit qunn Eigen3::Eigen)
add_test(
	NAME TestCircuit
	COMMAND TestCircuit
)

add_executable(TestGrad "Tests/TestGrad.cpp")
target_link_libraries(TestGrad qunn Eigen3::Eigen)
add_test(
	NAME TestGrad
	COMMAND TestGrad
)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
	DEPENDS TestCircuit TestHamEvol TestDiagonalHamEvol TestProductHamEvol 
	TestGrad)


add_executable(qaoa_tfi "examples/qaoa_tfi.cpp")
target_link_libraries(qaoa_tfi qunn Eigen3::Eigen)

add_executable(ti_tfi "examples/ti_tfi.cpp")
target_link_libraries(ti_tfi qunn Eigen3::Eigen)

add_executable(diagonal_product_tfi "examples/diagonal_product_tfi.cpp")
target_link_libraries(diagonal_product_tfi qunn Eigen3::Eigen)
