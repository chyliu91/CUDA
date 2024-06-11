## CUDA 内存组织简介

CUDA 中的设备内存分类和特征：

![](./imgs/cuda_mem.png)

![](./imgs/cuda_mem2.png)



### 全局内存

“全局内存”（global memory）的含义是核函数中的所有线程都能够访问其中的数据，全局内存由于没有存放在GPU的芯片上，因此具有较高的延迟和较低的访问速度。然而，它的容量是所有设备内存中最大的。其容量基本上就是显存容量。

全局内存的主要角色是为核函数提供数据，并在主机与设备及设备与设备之间传递数据。我们用cudaMalloc()函数为全局内存变量分配设备内存。然后，可以直接在核函数中访问分配的内存，改变其中的数据值。我们说过，要尽量减少主机与设备之间的数据传输，但有时是不可避免的。可以用cudaMemcpy()函数将主机的数据复制到全局内存，或者反过来。

全局内存可读可写。全局内存对整个网格的所有线程可见。也就是说，一个网格的所有线程都可以访问（读或写）传入核函数的设备指针所指向的全局内存中的全部数据。

全局内存的生命周期（lifetime）不是由核函数决定的，而是由主机端决定的。例如生命周期从 cudaMalloc() 开始，到主机端用cudaFree()释放它们的内存结束。在这期间，可以在相同的或不同的核函数中多次访问这些全局内存中的数据。

在处理逻辑上的两维或三维问题时，可以用cudaMallocPitch()和cudaMalloc3D()函数分配内存，用cudaMemcpy2D()和cudaMemcpy3D()复制数据，释放时依然用cudaFree()函数。

以上所有的全局内存都称为线性内存（linear memory）。在CUDA中还有一种内部构造对用户不透明的（not transparent）全局内存，称为CUDA Array。CUDA Array使用英伟达公司不对用户公开的数据排列方式，专为纹理拾取服务。

我们前面介绍的全局内存变量都是动态地分配内存的。在CUDA中允许使用静态全局内存变量，其所占内存数量是在编译期间就确定的。而且，这样的静态全局内存变量，静态全局内存变量必须在所有主机与设备函数外部定义，所以是一种“全局的静态全局内存变量”。这里，第一个“全局”的含义与C++中全局变量的含义相同，指的是对应的变量对从其定义之处开始、一个翻译单元内的所有设备函数直接可见。如果采用所谓的分离编译（separate compiling），还可以将可见范围进一步扩大。


静态全局内存变量由以下方式在任何函数外部定义：

- `_device_- Tx;`   单个变量
- `_device_- Ty[N];` 固定长度的数组


其中，修饰符 `_device_`  说明该变量是设备中的变量，而不是主机中的变量；`T` 是变量的类型；`N` 是一个整型常数。


在核函数中，可直接对静态全局内存变量进行访问，并不需要将它们以参数的形式传给核函数。不可在主机函数中直接访问静态全局内存变量，但可以用 cudaMemcpyToSymbol() 函数和 cudaMemcpyFromSymbol() 函数在静态全局内存与主机内存之间传输数据。这两个CUDA运行时API函数的原型如下：

```
cudaError_t cudaMemcpyToSymbol(

    const void* symbol, // 静态全局内存变量名
    const void* src,     // 主机内存缓冲区指针
    size_t count,        // 复制的字节数
    size_t offset = 0,   // 从symbol对应设备地址开始偏移的字节数
    cudaMemcpyKind kind = cudaMemcpyHostToDevice // 可选参数
);

cudaError_t cudaMemcpyFromSymbol(
    void* dst,          // 主机内存缓冲区指针
    const void* symbol,  // 静态全局内存变量名
    size_t count,        // 复制的字节数
    size_t offset = 0,   // 从symbol对应设备地址开始偏移的字节数
    cudaMemcpyKind kind = cudaMemcpyDeviceToHost // 可选参数
);

这
```






























