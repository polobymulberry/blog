---
title: CUDA学习系列(2) | 运行篇
date: 2019-04-06 23:15:33
updated: 2019-06-05 23:04:26
tags:
  - CUDA
categories:
  - CUDA
---

## 0x00 : 前言

上一篇主要学习了CUDA编译链接相关知识[CUDA学习系列(1) | 编译链接篇](https://polobymulberry.github.io/2019/03/04/CUDA%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97%281%29%20%7C%20%E7%BC%96%E8%AF%91%E9%93%BE%E6%8E%A5%E7%AF%87/)。了解编译链接相关知识可以解决很多CUDA编译链接过程中的疑难杂症，比如CUDA程序一启动就crash很有可能就是编译时候Real Architecture版本指定错误。当然，要真正提升CUDA程序的性能，就需要对CUDA本身的运行机制有所了解。顺便提及一下，CUDA有两个非常重要的特性，一个是**Thread Hierarchy**，主要是说CUDA运行时，其线程是如何分层次执行的，另一个是**Memory Hierarchy**，主要说CUDA显存是如何分层次进行分配和管理的。这篇文章主要阐述的是CUDA运行机制，也就是CUDA Thread Hierarchy，至于Memory Hierarchy则放在下一篇[CUDA学习系列(3) | 显存篇](https://polobymulberry.github.io/2019/05/26/CUDA%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97%283%29%20%7C%20%E6%98%BE%E5%AD%98%E7%AF%87/)进行详细介绍。

<!-- more -->

## 0x01 : Thread Hierarchy

CUDA的并行架构其实本质上就是SIMT(Single Instruction Multiple Threads)架构，其中SIMT和SIMD的区别可以参考这篇文章[【AI系统】SIMD & SIMT 与 CUDA 关系](https://zhuanlan.zhihu.com/p/5739361088)。SIMT架构体现了CUDA强大的并行能力，另外这种并行能力还具有层次性，更具体地说就是CUDA的线程分为三个层次，分别为Grid、Block、Thread。三者的关系可以用下图阐明：

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/grid_block_thread.png" width="50%" style="display: block; margin: 0 auto;">

简单点说CUDA将一个GPU设备抽象成了一个Grid，而每个Grid里面有很多Block，每个Block里面又会有很多Thread，最终由每个Thread去处理kernel函数。**这里其实有一个疑惑，每个device抽象成一个Grid还能理解，为什么不直接将Grid抽象成许多Thread呢，中间为什么要加一层Block抽象呢？**要能够回答这个问题，就不得不提CUDA的硬件架构，因为这种层次性的体现是和CUDA的硬件架构密不可分的。

## 0x02 : CUDA硬件架构

CUDA的硬件架构已经经历很很多代了，从Tesla到Volta，再到最近的Turning。具体Roadmap可以看下图：

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/GPU-Roadmap-GTC-2015-SGEMM.jpg" width="80%" style="display: block; margin: 0 auto;">

虽然随着架构升级，CUDA显卡的性能越来越高，但是核心架构还是之前的一套。比如拿Fermi和Turning架构对比下。


Fermi Architecture图如下：

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Fermi_Architecture.png" width="75%" style="display: block; margin: 0 auto;">

Turning Architecture图如下：

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Turning_Architecture.jpg" width="50%" style="display: block; margin: 0 auto;">

是不是感觉看起来Turning架构只是由很多Fermi架构组合在一起的，整体上看起来更复杂了一些，当然这种说法有很大问题，此处只是为了说明不同架构的虽然形式上不一样，但是本质其实差不多。所以为了方便学习，下面就以Fermi架构为主介绍CUDA硬件架构。


我们已经知道了一个GPU设备对应一个Grid。那么Block和Thread是如何和GPU里面的硬件对应上的呢？首先看看架构图中那些绿色区块部分，这是GPU架构中非常核心的一个部件，叫做SM(Streaming Multiprocessor)。下面看看Fermi SM的结构图：

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Fermi_SM.png" width="50%" style="display: block; margin: 0 auto;">

其中Core部分就是真正执行CUDA指令的地方，类似CPU Core。图中蓝色部分代表的是Memory，比如Shared Memory、L1 cache、Register File等。LD/ST是用来Load/Store Memory的。SFU是Special Fuction Unit，存储了内置的函数，比如sine、cosine等等，内置的函数要比标准函数快，但是精度低。另外就是橙色部分，主要是跟Wrap调度有关系，Wrap也是一个很重要的概念，后续会详细介绍。说到这里，基本上大家也能猜出来了Thread Hierarchy对应的硬件架构。

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Thread_Hierarchy_Software_Hardware.png" width="75%" style="display: block; margin: 0 auto;">

一个Device对应一个Grid，SM一次只执行一个Block，但是多个Block会在同一个SM中依次执行，CUDA Core一次只执行一个Thread，但是多个Thread可能会在同一个CUDA Core中依次执行。


为了让大家印象更加深刻，放上一张GPU的硬件图，看看是不是和上面的架构图很像。

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Turning_Hardware.jpg" width="80%" style="display: block; margin: 0 auto;">

至此，对于CUDA的Thread Hierarchy我们已经有了很清楚的认识了。至于blockIdx.xyz和threadIdx.xyz这些概念其实是从Software层面来说的，是为了方便不同类型数据的处理提出的线程模型，比如对于2D纹理处理，就适合2D Grid&2D Blocks。但是从硬件层面来说，不管是2D还是3D Blocks，其实对应的硬件模型都是一样的。这里就不赘述了，已经有很多优秀的文章介绍这些了。

## 0x03 : Warp

上一章节提到了Warp，但是没有细说。因为Warp本身是可以单独列一个主题来说的。理解Warp对优化CUDA程序有很重要的意义。**其实真正在运行过程中，每个SM同时执行的只有32个Thread，而这32个Thread统称为一个Warp**。如下图所示：

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Warp_Software_Hardware.png" width="80%" style="display: block; margin: 0 auto;">

也就是从开发者角度来看，block中所有的thread是同时运行的，但是真正每个block中同一时刻运行的thread只有32个。这里就有一个Warp调度的问题，所以每个SM中具备Warp Scheduler和Dispatch Unit用来调度不同的Warp。如下图所示：

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Warp_Scheduler.png" width="65%" style="display: block; margin: 0 auto;">

知道Warp概念后，就不得不提Warp Divergence。我们先看下面这段代码：

```cpp
__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
```

对于这段代码，在一个warp中，奇数位的thread会执行`b = 200.0f;`，而偶数位的thread会执行`a = 100.0f;`。如果是这段代码是执行在CPU上的，那么就会产生条件转移指令，不同分支的线程遇到该条件转移指令，会跳转到不同的位置继续执行指令，而且CPU端有复杂的分支预测器来减少分支跳转的开销。但是GPU上是没有这些复杂的分支处理机制的，所以GPU在执行时，warp中所有thread执行的指令是一样的，唯一不同的是，当遇到条件分支，如果满足该条件，就继续执行对应的指令，如果不满足该条件，该thread就会阻塞，直到其他满足该条件的thread执行完这段条件语句，上述现象就是Warp Divergence。上面这段话表达的可能不是很清楚，参考下图就比较明白了。

<img src="/2019/04/06/CUDA学习系列(2)%20%7C%20运行篇/Warp_Divergence.png" width="100%" style="display: block; margin: 0 auto;">

发生Warp Divergence的时候会造成性能的下降，所以针对上面代码，我们可以进行如下改造，奇数位的warp会执行`b = 200.0f;`，而偶数位的warp会执行`a = 100.0f;`：

```cpp
__global__ void mathKernel2(void) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
```

上面提到GPU没有复杂的分支处理机制，这是有原因的。CPU并行和GPU并行都是为了提升程序的性能。其中CPU注重的是Latency，所以CPU单核的性能非常强劲，这也造就了CPU的硬件机制比较复杂。而GPU注重的是Throughput，所以GPU单核的性能不是很强劲，但是核的数目要远远超过CPU，以此保证总的吞吐量远远超过CPU。

## 0x04 : 同步机制

CUDA的线程机制是分层的，对应的同步机制也分为两层：
1. System-Level，表示等待host和device的任务全部完成。对应函数为`cudaError_t cudaDeviceSynchronize(void);`
2. Block-Level，表示在device上等待同一个block中所有的thread执行到某个同步点。对应函数为**device** `void __syncthreads(void);`

对于Block-Level的同步，很容易出错。比如下面这段代码，就会造成阻塞，因为block同步的一个核心前提是**同一个block中所有的thread**执行到同一个点：

```cpp
__global__ void blockSync() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid % 2 == 0) {
        __syncthreads();
    } else {
        __syncthreads();
    }
}
```

## 0x05 : 参考文献

- [Single instruction, multiple threads](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)
- Professional CUDA C Programming
