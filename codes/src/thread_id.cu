#include <cstdio>

__global__ void hello_world()
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    printf("hello world from block:%d and thread:%d.\n", bid, tid);
}

int main()
{
    hello_world<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}