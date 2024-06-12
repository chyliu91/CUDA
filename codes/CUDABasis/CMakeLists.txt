cmake_minimum_required(VERSION 3.10)
enable_language(CUDA)

project(cuda_basis LANGUAGE CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

include_directories(./)

add_subdirectory(cuda_program)
add_subdirectory(performance_analysis_optimization)
add_subdirectory(threads)

