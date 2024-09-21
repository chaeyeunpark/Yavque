cmake_minimum_required(VERSION 3.14)

project(yavque)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)

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
target_link_libraries(yavque_dependency INTERFACE EDLIB::edlib)

# Load json
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(yavque_dependency INTERFACE nlohmann_json::nlohmann_json)

# Load eigen
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Setup BLAS
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	message(STATUS "Use Intel Compiler with natural mkl support")
	target_compile_options(yavque_dependency INTERFACE "-mkl=parallel -tbb")
	target_link_libraries(yavque_dependency INTERFACE tbb stdc++ pthread m dl)
	target_compile_options(yavque_dependency INTERFACE "-DEIGEN_USE_MKL_ALL") # For Eigen
else ()
    message(STATUS "For non Intel compiler, system BLAS will be used.")
    find_package(BLAS REQUIRED)

    message(STATUS "Using BLAS library with ${BLAS_LIBRARIES}.\n"
        "To use another BLAS implementation, set BLA_VENDOR variable in CMake.")

	target_link_libraries(yavque_dependency INTERFACE "${BLAS_LIBRARIES}")
	target_link_options(yavque_dependency INTERFACE "${BLAS_LINKER_FLAGS}")
	target_link_libraries(yavque_dependency INTERFACE tbb stdc++ pthread m dl)
	target_compile_options(yavque_dependency INTERFACE "-DEIGEN_USE_BLAS") # For Eigen
endif()

# Build executables
option(YAVQUE_BUILD_TESTS "Build tests" OFF)
option(YAVQUE_BUILD_EXAMPLES "Build examples" OFF)
option(YAVQUE_ENABLE_WARNINGS "Enable warnings" ON)

if(YAVQUE_ENABLE_WARNINGS)
    #Set -Wall -Wextra -Werror -Wpedantic
    target_compile_options(yavque_dependency INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>)
endif()

add_library(yavque STATIC "src/VariableImpl.cpp" "src/Circuit.cpp" 
	"src/Operators/SumPauliString.cpp" "src/Utilities/pauli_operators.cpp" 
	"src/backward_grad.cpp")
target_link_libraries(yavque PUBLIC Eigen3::Eigen yavque_dependency)
target_include_directories(yavque PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_include_directories(yavque PUBLIC 
	"${PROJECT_SOURCE_DIR}/externals/ExactDiagonalization/include/edlib/"
	"${PROJECT_SOURCE_DIR}/externals/nlohmann_json_cmake_fetchcontent/include/")

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
	add_subdirectory("${PROJECT_SOURCE_DIR}/Tests")
endif()
