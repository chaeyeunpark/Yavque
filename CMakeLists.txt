cmake_minimum_required(VERSION 3.9)

project(QUNN)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories("${CMAKE_SOURCE_DIR}/externals" "${CMAKE_SOURCE_DIR}/externals/ExactDiagonalization/Include/")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DEIGEN_USE_MKL_ALL")

## Setup MKL
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(mkl_intel_lp64 mkl_gnu_thread mkl_core gomp pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	include_directories("$ENV{MKLROOT}/include")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
	link_libraries(mkl_intel_lp64 mkl_gnu_thread mkl_core gomp pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(iomp5 pthread m dl)
endif()

add_library(qunn STATIC "src/VariableImpl.cpp" "src/Circuit.cpp" "src/utility.cpp")
target_link_libraries(qunn Eigen3::Eigen)

add_executable(TestHamEvol "Tests/TestHamEvol.cpp")
target_link_libraries(TestHamEvol qunn Eigen3::Eigen)

add_executable(TestCircuit "Tests/TestCircuit.cpp")
target_link_libraries(TestCircuit qunn Eigen3::Eigen)

add_executable(TestGrad "Tests/TestGrad.cpp")
target_link_libraries(TestGrad qunn Eigen3::Eigen)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
	DEPENDS TestHamEvol TestCircuit TestGrad)

add_executable(trotter "src/trotter.cpp")
target_link_libraries(trotter Eigen3::Eigen)
