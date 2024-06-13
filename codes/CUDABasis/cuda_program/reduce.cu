#include "error.cuh"
#include <stdio.h>

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(double) * N;
const int BLOCK_SIZE = 128;

void timing(double *h_x, double *d_x, const int method);

int main(void)
{
    double *h_x = (double *)malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    double *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CUDA_CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_global(double *d_x, double *d_y)
{
    const int tid = threadIdx.x;
    double *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}

void __global__ reduce_shared(double *d_x, double *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ double s_y[128];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

void __global__ reduce_dynamic(double *d_x, double *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ double s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

double reduce(double *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(double) * grid_size;
    const int smem = sizeof(double) * BLOCK_SIZE;
    double *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, ymem));
    double *h_y = (double *)malloc(ymem);

    switch (method)
    {
    case 0:
        reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
        break;
    case 1:
        reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
        break;
    case 2:
        reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
        break;
    default:
        printf("Error: wrong method\n");
        exit(1);
        break;
    }

    CUDA_CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    double result = 0.0;
    for (int n = 0; n < grid_size; ++n)
    {
        result += h_y[n];
    }

    free(h_y);
    CUDA_CHECK(cudaFree(d_y));
    return result;
}

void timing(double *h_x, double *d_x, const int method)
{
    double sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CUDA_CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}