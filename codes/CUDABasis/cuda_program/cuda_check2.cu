#include "error.cuh"
#include <cmath>
#include <cstdio>
#include <ctime>

const double EPSILON = 1.0e-5;

__global__ void add(const double *a, const double *b, double *c, int size)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
    {
        c[tid] = a[tid], b[tid];
    }
}

void check(const double *a, const double *b, const double *result, const int N)
{
    for (int i = 0; i < N; i++)
    {
        if (fabs(a[i] + b[i] - result[i]) > EPSILON)
        {
            printf("index %d calculation error\n", i);
        }
    }
}

int main(void)
{
    const int N = 200;
    const int M = sizeof(double) * N;

    // 申请主机端的内存
    double *h_x = (double *)malloc(M);
    double *h_y = (double *)malloc(M);
    double *h_z = (double *)malloc(M);

    // 产生随机数
    srand((unsigned int)time(NULL));
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = (double)rand() / (double)RAND_MAX;
        h_y[n] = (double)rand() / (double)RAND_MAX;
    }

    // 申请设备端的内存
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc((void **)&d_x, M));
    CUDA_CHECK(cudaMalloc((void **)&d_y, M));
    CUDA_CHECK(cudaMalloc((void **)&d_z, M));

    // 内存到显存拷贝
    CUDA_CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    // 定义执行配置, 故意将 block_size 设置大于上限 1024 的值
    const int block_size = 1025;
    const int grid_size = (N + block_size - 1) / block_size;

    // 执行核函数
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));

    check(h_x, h_y, h_z, N);

    // 释放主机端内存
    free(h_x);
    free(h_y);
    free(h_z);

    // 释放设备端的内存
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    printf("finished ...... \n");
    return 0;
}