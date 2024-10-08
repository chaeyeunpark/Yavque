cmake_minimum_required(VERSION 3.9)
project(yavque_tests)

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.6.0
)

FetchContent_MakeAvailable(Catch2)
get_target_property(Catch2_INC_DIR Catch2 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(Catch2 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Catch2_INC_DIR}")
# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(CTest)
include(Catch)

add_executable(runner TestBackwardGrad.cpp
                      TestCircuit.cpp
                      TestDiagonalHamEvol.cpp
                      TestGrad.cpp
                      TestHamEvol.cpp
                      TestSingleQubitOperator.cpp
                      TestSumLocalHamEvol.cpp
                      TestSumPauliString.cpp
                      TestSumPauliStringHamEvol.cpp
                      TestTFI.cpp
                      TestTwoQubitOperator.cpp)

target_link_libraries(runner PRIVATE yavque Eigen3::Eigen Catch2::Catch2WithMain EDLIB::edlib)

catch_discover_tests(runner)
