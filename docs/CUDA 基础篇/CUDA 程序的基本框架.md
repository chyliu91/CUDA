## CUDA 程序的基本框架

CUDA 程序的基本框架为:
```
头文件包含
常量定义（或者宏定义）
C++自定义函数和CUDA核函数的声明（原型）

int main(void)
{
    分配主机与设备内存
    初始化主机中的数据
    将某些数据从主机复制到设备
    调用核函数在设备中进行计算
    将某些数据从设备复制到主机
    释放主机与设备内存
}

C++自定义函数和CUDA核函数的定义（实现）
```

例如实现数组相加:
```

#include <cstdio>
#include <cmath>
#include <ctime>

const double EPSILON = 1.0e-5;

__global__ void add(const double* a, const double* b, double* c);

void check(const double *a, const double *b, const double* result, const int N);

int main(void) {
    const int N = 1e8; 
    const int M = sizeof(double) * N;

	// 申请主机端的内存
    double *h_x = (double *)malloc(M);
    double *h_y = (double *)malloc(M);
    double *h_z = (double *)malloc(M);

	// 产生随机数
	srand((unsigned int)time(NULL));
    for (int n = 0; n < N; ++n) {
        h_x[n] = (double)rand() / (double)RAND_MAX;
        h_y[n] = (double)rand() / (double)RAND_MAX;
    }

	// 申请设备端的内存
    double *d_x, *d_y, *d_z;
	cudaMalloc((void**)&d_x, M);
	cudaMalloc((void**)&d_y, M);
	cudaMalloc((void**)&d_z, M);

	// 内存到显存拷贝
	cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

	// 定义执行配置
    const int block_size = 128;
    const int grid_size = N / block_size;

	// 执行核函数
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

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


__global__ void add(const double* a, const double* b, double* c)
{
	const int tid = blockDim.x* blockIdx.x + threadIdx.x; 
	c[tid] =  a[tid] + b[tid];
}

void check(const double *a, const double *b, const double* result, const int N)
{
	for(int i = 0;i < N;i++)
	{
		if(fabs(a[i] + b[i] - result[i]) > 1e-15)
		{
			printf("index %d calculation error\n", i);
		}
	}
}
```

- 在 CUDA 运行时 API 中，没有明显地初始化设备的函数。在第一次调用一个和设备管理及版本查询功能无关的运行时 API 函数时，设备将自动初始化
- 所有的 CUDA 运行时 API 函数都以 `cuda` 开头


#### 内存分配

cuda 内存申请使用的 API:
```
cudaError_t cudaMalloc(void **address, size_t size);
```

- 第一个参数 `address` 是待分配设备内存的指针。注意：因为内存本身就是一个指针，所以待分配设备内存的指针就是指针的指针，即双重指针，例如，对于 `d_x`，该函数的功能是改变指针 `d_x` 本身的值（将一个指针赋值给 `d_x`），而不是改变 `d_x` 所指内存缓冲区中的变量值
- 第二个参数 `size` 是待分配内存的字节数
- 返回值是一个错误代号。如果调用成功，则返回 `cudaSuccess`；否则返回一个代表某种错误的代号


用 `cudaMalloc()` 函数分配的设备内存需要用 `cudaFree()` 函数释放。该函数的原型为:

```
cudaError_t cudaFree(void* address);
```

- 参数 `address` 就是待释放的设备内存变量（不是双重指针）
- 返回值是一个错误代号。如果调用成功，返回 `cudaSuccess`
- 在分配与释放各种内存时，相应的操作一定要两两配对，否则将有可能出现内存错误

### 主机与设备之间数据的传递


在分配了设备内存之后，就可以将某些数据从主机传递到设备中去了，这里用到了 CUDA 运行时 API 函数 `cudaMemcpy()`，其原型如下：
```
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
```
- 该函数的作用是将一定字节数的数据从源地址所指缓冲区复制到目标地址所指缓冲区
- 第一个参数 `dst` 是目标地址
- 第二个参数 `src` 是源地址
- 第三个参数 `count` 是复制数据的字节数
- 第四个参数 `kind` 是一个枚举类型的变量，标志数据传递方向。它只能取如下几个值：
    - `cudaMemcpyHostToHost`：从主机内存复制到主机内存
    - `cudaMemcpyHostToDevice`：从主机内存复制到设备内存
    - `cudaMemcpyDeviceToHost`：从设备内存复制到主机内存
    - `cudaMemcpyDeviceToDevice`：从设备内存复制到设备内存
    - `cudaMemcpyDefault`，表示根据指针 `dst` 和 `src` 所指地址自动判断数据传输的方向。这要求系统具有统一虚拟地址（unified virtual addressing）的功能（要求64位的主机）
- 返回值是一个错误代号。如果调用成功，则返回 `cudaSuccess`


### 核函数中数据与线程的对应关系

上面的示例代码中，使用具有 `128` 个线程的一维线程块，一共有 `1e8/128` 个这样的线程块。

在主机函数中，需要依次对数组的每一个元素进行操作，所以需要使用一个循环。在设备的核函数中，用“单指令多线程”的方式编写代码，故可去掉该循环，只需将数组元素下标与线程下标一一对应即可。

通过下面的计算确定了数组元素的下标：
```
const int n = blockDim.x * blockIdx.x + threadIdx.x;
```
从而：

- 第 `0` 个线程块中的 `blockDim.x` 个线程对应于第 `0` 个到第 `blockDim.x-1` 个数组元素
- 第 `1` 个线程块中的 `blockDim.x` 个线程对应于第 `blockDim.x` 个到第 `2*blockDim.x-1` 个数组元素，以此类推
- 核函数中定义的线程数目与数组元素数目一样，都是 `1e8`。在将线程下标与数据下标一一对应之后，就可以对数组元素进行操作了：
```
c[tid] =  a[tid] + b[tid];
```

在调试程序时，也可以仅仅使用一个线程。为此，可以先将核函数中的代码改成对应主机函数中的代码，然后用执行配置 `<<<1, 1>>>` 调用核函数。

### 核函数的要求

- 核函数的返回类型必须是 `void`。所以，在核函数中可以用 `return` 关键字，但不可返回任何值
- 必须使用限定符 `__global__`。也可以加上一些其他 C++ 中的限定符，如 `static`。限定符的次序可任意
- 函数名无特殊要求，而且支持 C++ 中的重载，即可以用同一个函数名表示具有不同参数列表的函数
- 不支持可变数量的参数列表，即参数的个数必须确定
- 可以向核函数传递非指针变量
- 除非使用统一内存编程机制，否则传给核函数的数组（指针）必须指向设备内存
- 核函数不可成为一个类的成员。通常的做法是用一个包装函数调用核函数，而将包装函数定义为类的成员
- 在计算能力 3.5 之前，核函数之间不能相互调用。从计算能力 3.5 开始，引入了动态并行机制，在核函数内部可以调用其他核函数，甚至可以调用自己（递归函数）
- 无论是从主机调用，还是从设备调用，核函数都是在设备中执行。调用核函数时必须指定执行配置，即三括号及其中的参数


### 核函数中 if 语句的必要性

上面的例子中我们使用的 block 数量为 `1e8/128=781250` 正好可以整除，那么如果计算的数组和分配的线程数量不是正好相等的关系呢？
例如将数组大小修改为：

```
const int N = 1e8+1; 
```
此时按照原来的 grid_size 计算方法:
```
const int block_size = 128;
const int grid_size = N / block_size;
```
显然最后一对数组相加没有给分配到线程。为了计算最后一对数组元素，我们可以分配 `781250+1` 个 block：
```
const int grid_size = (N - 1) / block_size + 1;
```
或者写为:
```
const int grid_size = (N + block_size - 1) / block_size;
```

以上两个语句都等价于下述语句：
```
int grid_size = (N % block_size == 0)
                ? (N / block_size)
                : (N / block_size + 1)
```
此时线程的数量 `1e8+128`多于数组元素个数 ·`1e8+1`，还按照原来的方式进行计算，会使得线程访问到它不应该访问的数据，因此，需要在核函数中加上判断：
```
__global__ void add(const double* a, const double* b, double* c, int size)
{
	const int tid = blockDim.x* blockIdx.x + threadIdx.x;
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
}
```

## 设备函数

核函数可以调用不带执行配置的自定义函数，这样的自定义函数称为设备函数（device function）。它是在设备中执行，并在设备中被调用的。

### 函数执行空间标识符

在 CUDA 程序中，由以下标识符确定一个函数在哪里被调用，以及在哪里执行：

- 用 `__global__` 修饰的函数称为核函数，一般由主机调用，在设备中执行。如果使用动态并行，则也可以在核函数中调用自己或其他核函数
- 用 `__device__` 修饰的函数称为设备函数，只能被核函数或其他设备函数调用，在设备中执行
- 用 `__host__` 修饰的函数就是主机端的普通 C++ 函数，在主机中被调用，在主机中执行。对于主机端的函数，该修饰符可省略。之所以提供这样一个修饰符，是因为有时可以用 `__host__` 和 `__device__` 同时修饰一个函数，使得该函数既是一个 C++ 中的普通函数，又是一个设备函数。这样做可以减少余代码。编译器将针对主机和设备分别编译该函数
- 不能同时用 `__device__` 和 `__global__` 修饰一个函数，即不能将一个函数同时定义为设备函数和核函数
- 也不能同时用 `__host__` 和 `__global__` 修饰一个函数，即不能将一个函数同时定义为主机函数和核函数 
- 编译器决定把设备函数当作内联函数或非内联函数，但可以用修饰符 `__noinline__` 建议一个设备函数为非内联函数（编译器不一定接受），也可以用修饰符 `__forceinline__` 建议一个设备函数为内联函数

### 使用设备函数
可以定义带返回值额设备函数

```
// 带返回值的设备函数
__device__ double dev_add1(const double a, const double b) {
	return a + b;
}


__global__ void add(const double* a, const double* b, double* c, int size)
{
	const int tid = blockDim.x* blockIdx.x + threadIdx.x;
	if (tid < size) {
		c[tid] = dev_add1(a[tid], b[tid]);
	}
}
```

也可以定义传递指针的设备函数：
```
// 传递指针的设备函数
__device__ void dev_add2(const double a, const double b, double* c) {
	*c = a + b;
}


__global__ void add(const double* a, const double* b, double* c, int size)
{
	const int tid = blockDim.x* blockIdx.x + threadIdx.x;
	if (tid < size) {
		dev_add2(a[tid], b[tid], &c[tid]);
	}
}
```

也可以定义传递引用的设备函数：
```
// 传递引用的设备函数
__device__ void dev_add3(const double a, const double b, double& c) {
	c = a + b;
}


__global__ void add(const double* a, const double* b, double* c, int size)
{
	const int tid = blockDim.x* blockIdx.x + threadIdx.x;
	if (tid < size) {
		dev_add3(a[tid], b[tid], c[tid]);
	}
}
```

## CUDA 程序的错误检查

### 定义错误检查宏

CUDA 运行时 API 返回的运行状态码可以用于错误检查，定义 `error.cuh`：
```
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
```

故意制造设备端计算完成后，数据从设备端拷贝到主机端时的错误：
```
#include <cstdio>
#include <cmath>
#include <ctime>
#include "error.cuh"


__global__ void add(const double* a, const double* b, double* c, int size)
{
	const int tid = blockDim.x* blockIdx.x + threadIdx.x;
	if (tid < size) {
		c[tid] = a[tid], b[tid];
	}
}

int main(void) {
	const int N = 200;
	const int M = sizeof(double) * N;

	// 申请主机端的内存
	double *h_x = (double *)malloc(M);
	double *h_y = (double *)malloc(M);
	double *h_z = (double *)malloc(M);

	// 产生随机数
	srand((unsigned int)time(NULL));
	for (int n = 0; n < N; ++n) {
		h_x[n] = (double)rand() / (double)RAND_MAX;
		h_y[n] = (double)rand() / (double)RAND_MAX;
	}

	// 申请设备端的内存
	double *d_x, *d_y, *d_z;
	CUDA_CHECK(cudaMalloc((void**)&d_x, M));
	CUDA_CHECK(cudaMalloc((void**)&d_y, M));
	CUDA_CHECK(cudaMalloc((void**)&d_z, M));

	// 内存到显存拷贝
	CUDA_CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

	// 定义执行配置
	const int block_size = 128;
	const int grid_size = (N + block_size - 1) / block_size;

	// 执行核函数
	add << <grid_size, block_size >> > (d_x, d_y, d_z, N);

	// 将结果从设备端拷回主机端，故意写成 cudaMemcpyHostToDevice
	CUDA_CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyHostToDevice));

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
```
输出:
```
File: add.cu, Line: 50, Error Code: 11, Error Text: invalid argument
```

### 检查核函数
用上述方法不能捕捉调用核函数的相关错误，因为核函数不返回任何值。

但是可以在核函数调用之后，加上下面两个语句来查看核函数的执行情况:
```
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize);
```
- 第一条语句的作用是捕提第二个语句之前的最后一个错误
- 第二条语句的作用是同步主机与设备。之所以要同步主机与设备是因为核函数的调用是异步的，即主机发出调用核函数的命令后会立即执行后面的语句，不会等待核函数执行完毕

需要注意的是，上述同步函数是比较耗时的，如果在程序的较内层循环调用的话，很可能会严重降低程序的性能。所以，一般不在程序的较内层循环调用上述同步函数。

故意制造核函数执行配置错误的程序：
```
#include <cstdio>
#include <cmath>
#include <ctime>
#include "error.cuh"


__global__ void add(const double* a, const double* b, double* c, int size)
{
	const int tid = blockDim.x* blockIdx.x + threadIdx.x;
	if (tid < size) {
		c[tid] = a[tid], b[tid];
	}
}

void check(const double *a, const double *b, const double* result, const int N)
{
	for (int i = 0; i < N; i++)
	{
		if (fabs(a[i] + b[i] - result[i]) > 1e-15)
		{
			printf("index %d calculation error\n", i);
		}
	}
}

int main(void) {
	const int N = 200;
	const int M = sizeof(double) * N;

	// 申请主机端的内存
	double *h_x = (double *)malloc(M);
	double *h_y = (double *)malloc(M);
	double *h_z = (double *)malloc(M);

	// 产生随机数
	srand((unsigned int)time(NULL));
	for (int n = 0; n < N; ++n) {
		h_x[n] = (double)rand() / (double)RAND_MAX;
		h_y[n] = (double)rand() / (double)RAND_MAX;
	}

	// 申请设备端的内存
	double *d_x, *d_y, *d_z;
	CUDA_CHECK(cudaMalloc((void**)&d_x, M));
	CUDA_CHECK(cudaMalloc((void**)&d_y, M));
	CUDA_CHECK(cudaMalloc((void**)&d_z, M));

	// 内存到显存拷贝
	CUDA_CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

	// 定义执行配置, 故意将 block_size 设置大于上限 1024 的值
	const int block_size = 1025;
	const int grid_size = (N + block_size - 1) / block_size;

	// 执行核函数
	add << <grid_size, block_size >> > (d_x, d_y, d_z, N);
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
```

输出:
```
File: add.cu, Line: 59, Error Code: 9, Error Text: invalid configuration argument
```

如果不加上上面错误的检查，则只知道最终 `check` 函数显示的结果不对，很难查找具体的错误原因。


