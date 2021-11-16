cmake_minimum_required(VERSION 3.14)

project(yavque)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)

message(STATUS "Using BLAS library with ${BLAS_LIBRARIES}.\n"
	"To use another BLAS implementation, set BLA_VENDOR variable in CMake.")


# Build executables
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

# Clang-tidy
option(CLANG_TIDY "Set Clang-Tidy" OFF)


if(CLANG_TIDY)
	message(STATUS "Use Clang-Tidy")
	set(CMAKE_CXX_CLANG_TIDY 
"clang-tidy;-header-filter=${PROJECT_SOURCE_DIR}/include;-extra-arg=-std=c++17;\
-checks=*,-modernize-use-trailing-return-type,\
-fuchsia-overloaded-operator,-fuchsia-multiple-inheritance,\
-fuchsia-default-arguments-*,-llvm-header-guard,\
-*-non-private-member-variables-in-classes;\
-warnings-as-errors=*,-hicpp-uppercase-literal-suffix,-cppcoreguidelines-avoid-magic-numbers,\
-readability-magic-numbers,-clang-analyzer-core.uninitialized.UndefReturn,\
-clang-analyzer-optin.portability.UnixAPI")
endif()

# Download CATCH
set(CATCH_URL "https://github.com/catchorg/Catch2/releases/download/v2.11.1/catch.hpp")
set(CATCH_DOWNLOAD_PATH "${PROJECT_SOURCE_DIR}/externals/catch.hpp")

if (NOT EXISTS "${CATCH_DOWNLOAD_PATH}")
	file(DOWNLOAD "${CATCH_URL}" "${CATCH_DOWNLOAD_PATH}")
endif()


if(NOT DEFINED ENV{TBBROOT})
	message(FATAL_ERROR "OneTBB or IntelTBB must be installed for this library. \
	If you have already installed TBB, check wheter the environment varialbles are corretly loaded.")
endif()

add_library(external_dependency INTERFACE)
target_include_directories(external_dependency INTERFACE "$ENV{MKLROOT}/include")

# Setup BLAS
if ((CMAKE_CXX_COMPILER_ID STREQUAL "Intel") AND ($"{BLA_VENDOR}" MATCHES "Intel.*"))
	message(STATUS "Use Intel Compiler with natural mkl support")
	target_compile_options(external_dependency INTERFACE "-mkl=parallel -tbb")
	target_link_libraries(external_dependency INTERFACE tbb stdc++ pthread m dl)
	target_compile_options(external_dependency INTERFACE "-DEIGEN_USE_MKL_ALL") # For Eigen
else ()
	target_link_libraries(external_dependency INTERFACE	"${BLAS_LIBRARIES}")
	target_link_options(external_dependency INTERFACE "${BLAS_LINKER_FLAGS}")
	target_link_libraries(external_dependency INTERFACE tbb stdc++ pthread m dl)
	target_compile_options(external_dependency INTERFACE "-DEIGEN_USE_BLAS") # For Eigen
endif()

# Always set -Wall -Wextra -Werror -Wpedantic
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic")


add_library(yavque STATIC "src/VariableImpl.cpp" "src/Circuit.cpp" 
	"src/Operators/SumPauliString.cpp" "src/Utilities/pauli_operators.cpp" 
	"src/backward_grad.cpp")
target_link_libraries(yavque PUBLIC Eigen3::Eigen external_dependency)
target_include_directories(yavque PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_include_directories(yavque PUBLIC 
	"${PROJECT_SOURCE_DIR}/externals/ExactDiagonalization/Include/"
	"${PROJECT_SOURCE_DIR}/externals/nlohmann_json_cmake_fetchcontent/include/")

if(BUILD_EXAMPLES)
	add_executable(qaoa_tfi_diag "examples/qaoa_tfi_diag.cpp")
	target_link_libraries(qaoa_tfi_diag PRIVATE yavque Eigen3::Eigen)

	add_executable(qaoa_tfi_pauli "examples/qaoa_tfi_pauli.cpp")
	target_link_libraries(qaoa_tfi_pauli PRIVATE yavque Eigen3::Eigen)

	add_executable(qaoa_tfi_ti "examples/qaoa_tfi_ti.cpp")
	target_link_libraries(qaoa_tfi_ti PRIVATE yavque Eigen3::Eigen)

	add_executable(qaoa_cluster "examples/qaoa_cluster.cpp")
	target_link_libraries(qaoa_cluster PRIVATE yavque Eigen3::Eigen)

	add_executable(solve_chem "examples/solve_chem.cpp")
	target_link_libraries(solve_chem PRIVATE yavque Eigen3::Eigen)
endif()

if(BUILD_TESTS)
	enable_testing()
	add_subdirectory("${PROJECT_SOURCE_DIR}/Tests")
endif()
