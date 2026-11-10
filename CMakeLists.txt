cmake_minimum_required(VERSION 3.12)
project(MLE_For_iPCA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ipca
    ipca_main.cpp
    feasibility.cpp
    observed_likelihood.cpp
    structured_observed_likelihood.cpp
    flipflop.cpp
    quiver.cpp)

include(CTest)
if(BUILD_TESTING)
    add_executable(ipca_tests
        tests.cpp
        feasibility.cpp
        observed_likelihood.cpp
        structured_observed_likelihood.cpp
        flipflop.cpp
        quiver.cpp)
    add_test(NAME ipca_tests COMMAND ipca_tests)
endif()
