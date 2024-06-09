cmake_minimum_required(VERSION 3.12)
project(cuda_basis)

find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL "could not find CUDA")
else()
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(./)

function(add_cuda_executable name srcs)
    cuda_add_executable(${name} ${srcs})
    target_link_libraries(${name} ${CUDA_LIBRARIES})
    install(TARGETS ${name} DESTINATION bin)
endfunction()
    

add_subdirectory(cuda_program)
add_subdirectory(threads)

