#include <cstdio>

__global__ void hello_world()
{
    int bidx = blockIdx.x; // 0 ~ 1
    int bidy = blockIdx.y; // 0 ~ 2
    int bidz = blockIdx.z; // 0 ~ 1

    int tidx = threadIdx.x; // 0 ~ 2
    int tidy = threadIdx.y; // 0 ~ 1
    int tidz = threadIdx.z; // 0

    int gbid = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;       // 0~11
    int ltid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 0~5
    int gtid = gbid * blockDim.x * blockDim.y * blockDim.z + ltid;                             // 0~71

    printf("hello world from block:(%d, %d，%d) and thread:(%d, %d，%d)，with global block id:%d, local thread id:%02d, global thread id: %03d.\n",
           bidx, bidy, bidz, tidx, tidy, tidz, gbid, ltid, gtid);
}

int main()
{
    const dim3 grid_Size(2, 3, 2);
    const dim3 block_size(3, 2);
    hello_world<<<grid_Size, block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}