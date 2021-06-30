cmake_minimum_required(VERSION 3.14)

project(yavque)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

# download CATCH
set(CATCH_URL "https://github.com/catchorg/Catch2/releases/download/v2.11.1/catch.hpp")
set(CATCH_DOWNLOAD_PATH "${PROJECT_SOURCE_DIR}/externals/catch.hpp")

if (NOT EXISTS "${CATCH_DOWNLOAD_PATH}")
	file(DOWNLOAD "${CATCH_URL}" "${CATCH_DOWNLOAD_PATH}")
endif()


# set external dependencies
if(NOT DEFINED ENV{MKLROOT})
	message(FATAL_ERROR "MKL (which is not a part of OneAPI) must be installed for this library. \
	If you have already installed MKL, check wheter the environment varialbles are corretly loaded.")
endif()
if(NOT DEFINED ENV{TBBROOT})
	message(FATAL_ERROR "TBB (which is not a part of OneAPI) must be installed for this library. \
	If you have already installed TBB, check wheter the environment varialbles are corretly loaded.")
endif()

add_library(external_dependency INTERFACE)
target_include_directories(external_dependency INTERFACE "$ENV{MKLROOT}/include")

# setup MKL
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	target_compile_options(external_dependency INTERFACE "-mkl=parallel -tbb")
	target_link_libraries(external_dependency INTERFACE tbb stdc++ pthread m dl)
else()
	target_link_libraries(external_dependency INTERFACE	mkl_intel_lp64 
		mkl_tbb_thread mkl_core tbb pthread m dl)
endif()

# always set -Wall -Wextra
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")


add_library(yavque STATIC "src/VariableImpl.cpp" "src/Circuit.cpp" 
	"src/Operators/SumPauliString.cpp" "src/Utilities/pauli_operators.cpp" 
	"src/backward_grad.cpp")
target_link_libraries(yavque PUBLIC Eigen3::Eigen external_dependency)
target_include_directories(yavque PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_options(yavque PUBLIC "-DEIGEN_USE_MKL_ALL")

target_include_directories(yavque PUBLIC 
	"${PROJECT_SOURCE_DIR}/externals/ExactDiagonalization/Include/"
	"${PROJECT_SOURCE_DIR}/externals/nlohmann_json_cmake_fetchcontent/include/")

if(BUILD_EXAMPLES)
	add_executable(qaoa_tfi "examples/qaoa_tfi.cpp")
	target_link_libraries(qaoa_tfi PRIVATE yavque Eigen3::Eigen)

	add_executable(ti_tfi "examples/ti_tfi.cpp")
	target_link_libraries(ti_tfi PRIVATE yavque Eigen3::Eigen)

	add_executable(diagonal_product_tfi "examples/diagonal_product_tfi.cpp")
	target_link_libraries(diagonal_product_tfi PRIVATE yavque Eigen3::Eigen)

	add_executable(qaoa_cluster "examples/qaoa_cluster.cpp")
	target_link_libraries(qaoa_cluster PRIVATE yavque Eigen3::Eigen)
endif()

if(BUILD_TESTS)
	enable_testing()
	add_subdirectory("${PROJECT_SOURCE_DIR}/Tests")
endif()
