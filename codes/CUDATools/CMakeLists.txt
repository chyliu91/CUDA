cmake_minimum_required(VERSION 3.10)
enable_language(CUDA)

project(cuda_basis)

include(CheckLanguage)
checkLanguage(CUDA)

include_directories(./)



add_subdirectory(mem-check)


