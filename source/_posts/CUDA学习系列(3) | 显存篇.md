---
title: CUDA学习系列(3) | 显存篇 - 未完待续
date: 2019-05-26 16:22:12
updated: 2019-06-09 00:53:22
tags:
  - CUDA
categories:
  - CUDA
---

## 0x00 : 前言

上一篇文章[CUDA学习系列(2) | 运行篇](https://polobymulberry.github.io/2019/04/06/CUDA%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97%282%29%20%7C%20%E8%BF%90%E8%A1%8C%E7%AF%87/)主要介绍了CUDA的Thread Hierarchy，本文则介绍CUDA另一个重要特性Memory Hierarchy，意思是CUDA Memory有很多种类，这些种类可以根据层次进行划分。事实上，Thread Hierarchy和Memory Hierarchy中的层次性讲的是同一个东西，本质上都是由硬件的层次性决定的。除了显存的层次性外，显存中不同类型的Memory也是眼花缭乱。针对不同的情况选择不同的Memory，会对CUDA的运行有巨大影响。

<!-- more -->

## 0x01 : Memory Hierarchy

CUDA之所以要提出Memory层次性的概念，是基于一个共识：显存中有很多Memory种类，这些Memory都有一个这样的特性——**空间越大，延迟越大**，所以通过区分不同的层次，尽量满足不同的存储需求。另一方面，基于存储的局部性原理，Memory Hierarchy可以在少量的存储空间前提下，最大化存储性能。

<img src="/2019/05/26/CUDA学习系列(3)%20%7C%20显存篇/Memory_Latency_Capacity.png" width="50%" style="display: block; margin: 0 auto;">

上图表达的是大部分计算机中的存储层次，所以说[Memory hierarchy](https://en.wikipedia.org/wiki/Memory_hierarchy)并不是特别针对CUDA提出来的，而是大部分计算机通用的概念，只是CUDA Memory Hierarchy稍微有些特别而已。下图表示的就是CUDA中的Memory机制。

<img src="/2019/05/26/CUDA学习系列(3)%20%7C%20显存篇/Memory_Hierarchy.png" width="65%" style="display: block; margin: 0 auto;">

首先看看Device内部的Memory结构。可以看出有很多Memory类型，但是主要还是分为三种，也就是三个层次。


1.**Thread层次的Memory：**Register和Local Memory。
2.**Block层次的Memory：**Shared Memory，当然还有一些Cache Memory，图中没有提到，后面会简单介绍一下。
3.**Grid层次的Memory：**Global Memory、Constant Memory和Texture Memory。根据之前提到的Memory特性，不难知道，以上Memory空间从大到小排序依次是Grid Level Memory -> Block Level Memory -> Thread Level Memory，这也意味着延迟越来越小。当然凡事都有特例，比如Local Memory，虽然处于Thread，但是延迟非常大。下面针对这些不同类型的Memory做个简单介绍。

### Thread Memory

Thread Memory顾名思义就是每个Thread可以单独读写，但是其他Thread访问不了的Memory，也就是Thread私有的Memory。

1. **Register**
Register是CUDA Memory中最快的Memory类型。一般在kernel函数中声明的变量或者数组会放到Register中，当然数组必须是编译期就能确定大小的。Register的数目非常有限，并且是根据actived warps数目划分的，所以减少Register的使用可以提高actived warp的数目，进而提高性能。下图对这一现象进行了说明。

<img src="/2019/05/26/CUDA学习系列(3)%20%7C%20显存篇/More_Threads_With_Fewer_Register_Per_Thread.png" width="75%" style="display: block; margin: 0 auto;">

开发者想设置每个thread分配到的Register的数目有两种方法：
a. 给kernel函数添加<strong>launch_bounds</strong>声明，这样就可以在编译期间计算出每个thread所需的Register数目

```cpp
__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
kernel(...) {
    // your kernel body
}
```

b. 编译时指定`-maxrregcount=<n>`，表示所有kernel函数分配到的Register最多只有N个
想深入这两种方法的具体用法可以参考CUDA文档。

2. **Local Memory**
本来应该放在Register的数据由于空间不够了怎么办？这时候就会发生Register Spilling现象。会将本该放在Register的数据放到Local Memory中。但是Local Memory其实和Global Memory的物理空间是一样的，所以Local Memory和Global Memory一样具有低延迟的问题。

### Block Memory

Block Memory其实是由Shared Memory和各种Cache组成的。见下图。

<img src="/2019/05/26/CUDA学习系列(3)%20%7C%20显存篇/Block_Memory.png" width="50%" style="display: block; margin: 0 auto;">

1. **Shared Memory**
Shared Memory是显存中非常重要的一个概念。它是On-Chip Memory，比Global Memory和Local Memory延迟要小很多。只有同一个Block内的thread才能访问到同一个Shared Memory变量。
在kernel函数内部定义一个Shared Memory变量需要在其前面加上<strong>\_\_shared\_\_</strong>。当然你也可以在kernel函数外面定义<strong>\_\_shared\_\_</strong>变量，这代表所有kernel可见。当然你也可以使用extern预先声明一个<strong>\_\_shared\_\_</strong>变量。比如

```cpp
extern __shared__ int tile[];
```

注意上述声明的tile数组并没有指定数组大小。可以通过kernel的launch参数进行动态指定，目前只支持一维数组。

```cpp
// <<<>>>中第三个参数指的就是分配的Shared Memory大小
kernel<<<grid, block, isize * sizeof(int)>>>(...)
```

Shared Memory可以用作：
- Block内部thread之间的通信
- 作为Global Memory的Program-Managed Cache，提高Global Memory读取性能

2. **L1 Cache**
L1 Cache通常和L2 Cache一起用，通常是用来缓存Global Memory或者Local Memory的读取数据。在SM上，Shared Memory和L1 Cache用的是同一个On-Chip Memory，所以有下面这个函数对Shared Memory和L1 Cache的大小进行配置。

```cpp
/**
 * cudaFuncCachePreferNone: no preference (default)
 * cudaFuncCachePreferShared: prefer 48KB shared memory and 16KB L1 cache
 * cudaFuncCachePreferL1: prefer 48KB L1 cache and 16KB shared memory
 * cudaFuncCachePreferEqual: Prefer equal size of L1 cache and shared memory, both 32KB
 */
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
```

L1 Cache是可以通过编译选项将其关闭/打开的。

```bash
# disable L1 Cache
-Xptxas -dlcm=cg
# enable L1 Cache
-Xptxas -dlcm=ca
```

3. **Read Only Texture Cache**
Read Only Texture Cache是用来缓存Texture Memory的。它有一个强大的特性就是支持硬件级别的插值，对于2D纹理这类数据会有很大的用处。

4. **Read Only Constant Cache**
Read Only Constant Cache是用来缓存Constant Memory的。相对于Texture Cache，虽然都是Read Only，但是Constant Cache更适合Uniform Access，而Texture Cache更适合Scattered Access。

### Grid Memory

Grid Memory类似Host端的主存，其实它也是唯一和Host端的主存进行转移的显存。

1. **Global Memory**
Global Memory应该是平常开发过程中打交道最多的一种Memory类型，最大特点就是空间大但是延迟高。Global Memory变量使用<strong>\_\_device\_\_</strong>进行声明。虽然Global Memory本质上没有什么特殊的，但是Global Memory的使用和管理方面需要注意的点非常多，下一小节会详细讨论。

2. **Constant Memory**
对于一些常数非常适合放在Constant Memory中。并且Constant Memory变量需要使用<strong>\_\_constant\_\_</strong>进行声明。整个Constant Memory变量的使用可以参考如下代码：

```cpp
// f(x) = a0*x0+a1*x1+a2*x2+a3*x3+a4*x4
__constant__ float coef[5];

// 使用Constant Memory设置多项式的系数
void setup_coef_constant(void) {
    const float h_coef[] = {a0, a1, a2, a3, a4};
    // cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);
    cudaMemcpyToSymbol(coef, h_coef, (5) * sizeof(float));
}
```

3. **Texture Memory**
Texture Memory是一种特殊类型的Global Memory。但是Texture Memory顾名思义，对于2D存储的数据有很好的空间局部性。

4. **L2 Cache**
L2 Cache也是用来做缓存的，直接缓存Global Memory的数据。和L1 Cache不同的在于，L2 Cache是Per-Device Cache，而L1 Cache是Per-SM Cache。

## 0x02 : Memory Management

### Global Memory

### Shared Memory

## 0x03 : 参考文献

- Professional CUDA C Programming
- [Memory hierarchy](https://en.wikipedia.org/wiki/Memory_hierarchy)
