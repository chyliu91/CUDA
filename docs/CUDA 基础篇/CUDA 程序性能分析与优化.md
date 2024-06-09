## CUDA 程序性能分析
在 CUDA 程序中可以采用事件来记时:

```
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));
CUDA_CHECK(cudaEventRecord(start));
cudaEventQuery(start);

// 需要计时的代码快

CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));
float elapsed_time;

CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Cost Time: %g ms.", elapsed_time);

CUDA_CHECK(cudaEventDestroy(start));
CUDA_CHECK(cudaEventDestroy(stop));
```

- `cudaEventQuery(start);` 对处于 TCC 驱动模式的 GPU 来说可以省略，但对处于 WDDM 驱动模式的 GPU 来说必须保留。这是因为，在处于 WDDM 驱动模式的 GPU 中，一个 CUDA 流（CUDAstream）中的操作（如这里的 `cudaEventRecord()`函数）并不是直接提交给 GPU 执行，而是先提交到一个软件队列，需要添加一条对该流的 `cudaEventQuery()` 操作（或者 `cudaEventSynchronize()`）刷新队列，才能促使前面的操作在 GPU 执行
- `// 需要计时的代码快` 可以是一段主机代码（如对一个主机函数的调用），也可以是一段设备代码（如对一个核函数的调用），还可以是一段混合代码

例如，对向量相加的程序计时:
```

#include <cmath>
#include <cstdio>
#include <ctime>
#include "error.cuh"

const double EPSILON = 1.0e-5;

__global__ void add(const double *a, const double *b, double *c, int size)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size)
	{
		c[tid] = a[tid] + b[tid];
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

	// 申请设备端的内存
	double *d_x, *d_y, *d_z;
	cudaMalloc((void **)&d_x, M);
	cudaMalloc((void **)&d_y, M);
	cudaMalloc((void **)&d_z, M);

	// 内存到显存拷贝
	cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

	// 定义执行配置
	const int block_size = 128;
	const int grid_size = (N + block_size - 1) / block_size;
	printf("grid_size: %d\n", grid_size);

	float time_sum = 0.0;
	for (int i = 0; i < NUM_REPEATS; i++) {
		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));
		CUDA_CHECK(cudaEventRecord(start));
		cudaEventQuery(start);

		// 执行核函数
		add << <grid_size, block_size >> > (d_x, d_y, d_z, N);

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

	// 将结果从设备端拷回主机端
	cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

	check(h_x, h_y, h_z, N);

	// 释放主机端内存
	free(h_x);
	free(h_y);
	free(h_z);

	// 释放设备端的内存
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	printf("finished ...... \n");

	return 0;
}
```

我们忽略第一次测得的时间，因为第一次计算时，机器（无论是CPU还是GPU）都可能处于预热状态，测得的时间往往偏大。

### 数据传输的比例

GPU 和 CPU 之间的数据传输有可能是整个程序性能的瓶颈，要获得可观的GPU加速，就必须尽量缩减数据传输所花时间的比例。有时，即使有些计算在GPU中的速度并不高，也要尽量在GPU中实现，避免过多的 GPU 和 CPU 之间的数据传递。


### 算术强度
一个计算问题的算术强度指的是其中算术操作的工作量与必要的内存操作的工作量之比。例如，在数组相加的问题中，在对每一对数据进行求和时需要先将一对数据从设备内存中取出来，然后对它们实施求和计算，最后将计算的结果存放到设备内存。这个问题的算术强度其实是不高的，因为在取两次数据、存一次数据的情况下只做了一次求和计算。在 CUDA 中，设备内存的读、写都是代价高昂（比较耗时）的。

对设备内存的访问速度取决于 GPU 的显存带宽。以 GeForce RTX 2070 为例，其显存带宽理论值为 448GB/s。相比之下，该 GPU 的单精度浮点数计算的峰值性能为 6.5TFLOPS，意味着该 GPU 的理论寄存器带宽（只考虑浮点数运算，不考虑能同时进行的整数运算）为：

$$
\frac{4B(float 类型数据) \times 4(每个 FMA 的操作数)}{2(每个 FMA 浮点数的操作次数)} * 6.5*\frac{10^{12}}{s} =52 TB/s
$$

- FMA 指 fused multiply-add 指令，即涉及 4 个操作数和 2 个浮点数操作的运算 $d=α×b+c$
- 由此可见，对单精度浮点数来说，该GPU中的数据存取比浮点数计算慢 100 多倍