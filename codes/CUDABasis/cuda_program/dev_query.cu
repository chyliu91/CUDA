
#include <cuda_runtime.h>
#include <stdio.h>
 
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
 
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
 
        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Clock rate: %d\n", deviceProp.clockRate);
        printf("  Memory Clock rate: %d\n", deviceProp.memoryClockRate);
        printf("  Memory Bus Width: %d\n", deviceProp.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e9);
    }
 
    return 0;
}