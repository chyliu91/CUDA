#pragma once

#include <cstdio>

#define CUDA_CHECK(call)                                                                                                                        \
    do                                                                                                                                          \
    {                                                                                                                                           \
        const cudaError_t error_code = call;                                                                                                    \
        if (error_code != cudaSuccess)                                                                                                          \
        {                                                                                                                                       \
            printf("CUDA Error:\n");                                                                                                            \
            printf("	File: %s, Line: %d, Error Code: %d, Error Text: %s\n", __FILE__, __LINE__, error_code, cudaGetErrorString(error_code)); \
            exit(1);                                                                                                                            \
        }                                                                                                                                       \
    } while (0)