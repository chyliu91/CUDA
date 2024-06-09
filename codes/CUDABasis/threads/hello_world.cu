#include <cstdio>

__global__ void hello_world(){
	printf("hello world from gpu.\n");
}

int main()
{
	hello_world<<<1,1>>>();
	cudaDeviceSynchronize();

	return 0;
}