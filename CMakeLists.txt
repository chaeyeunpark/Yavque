cmake_minimum_required(VERSION 3.9)

project(yavque)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("${PROJECT_SOURCE_DIR}/externals") 
include_directories("${PROJECT_SOURCE_DIR}/externals/ExactDiagonalization/Include/")
include_directories("${PROJECT_SOURCE_DIR}/externals/nlohmann_json_cmake_fetchcontent/include/")
include_directories("${PROJECT_SOURCE_DIR}/externals/spectra/include/")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DEIGEN_USE_MKL_ALL")
set(CATCH_URL "https://github.com/catchorg/Catch2/releases/download/v2.11.1/catch.hpp")
set(CATCH_DOWNLOAD_PATH "${PROJECT_SOURCE_DIR}/externals/catch.hpp")


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


add_library(qunn STATIC "src/VariableImpl.cpp" "src/Circuit.cpp" "src/Operators/SumPauliString.cpp" "src/Utilities/pauli_operators.cpp")
target_link_libraries(qunn INTERFACE Eigen3::Eigen)

if(BUILD_EXAMPLES)
	add_executable(qaoa_tfi "examples/qaoa_tfi.cpp")
	target_link_libraries(qaoa_tfi PRIVATE qunn Eigen3::Eigen)

	add_executable(ti_tfi "examples/ti_tfi.cpp")
	target_link_libraries(ti_tfi PRIVATE qunn Eigen3::Eigen)

	add_executable(diagonal_product_tfi "examples/diagonal_product_tfi.cpp")
	target_link_libraries(diagonal_product_tfi PRIVATE qunn Eigen3::Eigen)

	add_executable(qaoa_cluster "examples/qaoa_cluster.cpp")
	target_link_libraries(qaoa_cluster PRIVATE qunn Eigen3::Eigen)
endif()

if(BUILD_TESTS)
	enable_testing()
	add_subdirectory("${PROJECT_SOURCE_DIR}/Tests")
endif()
