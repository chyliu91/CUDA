#include "error.cuh"
#include <cmath>
#include <cstdio>
#include <ctime>

const double EPSILON = 1.0e-5;

void add(const double *a, const double *b, double *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
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
    const int N = 1e8;
    const int M = sizeof(double) * N;
    const int NUM_REPEATS = 10;

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

    float time_sum = 0.0;
    for (int i = 0; i < NUM_REPEATS; i++)
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // 执行核函数
        add(h_x, h_y, h_z, N);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_time = 0.0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Round %d, Cost Time: %g ms.\n", i, elapsed_time);
        if (i > 0)
        {
            time_sum += elapsed_time;
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("Average Cost Time: %.3f\n", (float)time_sum / (NUM_REPEATS - 1));

    check(h_x, h_y, h_z, N);

    // 释放主机端内存
    free(h_x);
    free(h_y);
    free(h_z);

    printf("finished ...... \n");

    return 0;
}
