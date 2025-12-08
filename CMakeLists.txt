cmake_minimum_required(VERSION 3.14)

project(yavque)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_blas_version.cmake")

############################################################
#                  Setup Dependencies
############################################################
add_library(yavque_dependency INTERFACE)

# Load edlib
include(FetchContent)
FetchContent_Declare(
    edlib
    GIT_REPOSITORY https://github.com/chaeyeunpark/ExactDiagonalization.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(edlib)
get_target_property(edlib_IID edlib INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(edlib PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${edlib_IID}")
target_link_libraries(yavque_dependency INTERFACE EDLIB::edlib)

# Load json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP 1
)
FetchContent_MakeAvailable(json)
target_link_libraries(yavque_dependency INTERFACE nlohmann_json::nlohmann_json)

# Load eigen
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Load BLAS
message(STATUS "Finding BLAS")
find_package(BLAS REQUIRED)
get_blas_version(BLAS_VERSION)

if(BLAS_VERSION MATCHES "MKL")
    message(STATUS "Use MKL. Set EIGEN_USE_BLAS=1 for Eigen.")
    target_compile_options(yavque_dependency INTERFACE "-D EIGEN_USE_BLAS")
endif()

target_link_libraries(yavque_dependency INTERFACE BLAS::BLAS)

message(STATUS "Using BLAS library with ${BLAS_VERSION}. "
    "To use another BLAS implementation, set BLA_VENDOR variable in CMake.")

# Build executables
option(YAVQUE_BUILD_TESTS "Build tests" OFF)
option(YAVQUE_BUILD_EXAMPLES "Build examples" OFF)
option(YAVQUE_ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)

if(YAVQUE_ENABLE_WARNINGS)
    #Set -Wall -Wextra -Werror -Wpedantic
    target_compile_options(yavque_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>)
endif()

if(ENABLE_CLANG_TIDY)
    if(NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy-20)
    endif()
    message(STATUS "Using CLANG_TIDY_BINARY=${CLANG_TIDY_BINARY}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                             -extra-arg=-std=c++20;
    )
endif()

add_library(yavque STATIC "src/VariableImpl.cpp" "src/Circuit.cpp" 
	"src/Operators/SumPauliString.cpp" "src/Utilities/pauli_operators.cpp" 
	"src/backward_grad.cpp")
target_link_libraries(yavque PUBLIC Eigen3::Eigen yavque_dependency)
target_include_directories(yavque PUBLIC "${PROJECT_SOURCE_DIR}/include")

if(YAVQUE_BUILD_EXAMPLES)
	add_executable(qaoa_tfi_diag "examples/qaoa_tfi_diag.cpp" "examples/example_utils.cpp")
	target_link_libraries(qaoa_tfi_diag PRIVATE yavque Eigen3::Eigen)

	add_executable(qaoa_tfi_pauli "examples/qaoa_tfi_pauli.cpp" "examples/example_utils.cpp")
	target_link_libraries(qaoa_tfi_pauli PRIVATE yavque Eigen3::Eigen)

	add_executable(qaoa_tfi_ti "examples/qaoa_tfi_ti.cpp" "examples/example_utils.cpp")
	target_link_libraries(qaoa_tfi_ti PRIVATE yavque Eigen3::Eigen)

	add_executable(qaoa_cluster "examples/qaoa_cluster.cpp" "examples/example_utils.cpp")
	target_link_libraries(qaoa_cluster PRIVATE yavque Eigen3::Eigen)

	add_executable(solve_chem "examples/solve_chem.cpp" "examples/example_utils.cpp")
	target_link_libraries(solve_chem PRIVATE yavque Eigen3::Eigen)
endif()

if(YAVQUE_BUILD_TESTS)
	enable_testing()
	add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
endif()
