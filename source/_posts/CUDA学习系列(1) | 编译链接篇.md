---
title: CUDA学习系列(1) | 编译链接篇
date: 2019-03-04 00:52:22
updated: 2019-05-26 13:43:18
tags:
  - CUDA
categories:
  - CUDA
---

## 0x00 : 前言

CUDA的编译链接其实只需要看官方文档[cuda-compiler-driver-nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)即可。本文主要是在此官方文档的基础上，对其内容进行了补充说明，并增加了一些我个人的理解。


官方文档中总共有四张图片，我个人认为只要理解透了这四张图片，对CUDA编译链接的理解基本上在日常开发中够用了。下面我会结合这四张图片详细谈谈我对它们的理解。

<!-- more -->

## 0x01: 整体流程概述

假设当前我们的CUDA工程中有三个文件，分别为x.cu、y.cu、y.h，其中x.cu对y.h/cu中定义的device变量和函数进行了调用。整个代码参考的[cuda-compiler-driver-nvcc/examples](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples)里面内容。

```cpp
//---------- y.h ----------
#define N 8

extern __device__ int g[N];
extern __device__ void bar(void);
```

```cpp
//---------- y.cu ----------
#include "y.h"

__device__ int g[N];

__device__ void bar (void)
{
  g[threadIdx.x]++;
}
```

```cpp
//---------- x.cu ----------
#include <stdio.h>
#include "y.h"

__global__ void foo (void) {

  __shared__ int a[N];
  a[threadIdx.x] = threadIdx.x;

  __syncthreads();

  g[threadIdx.x] = a[blockDim.x - threadIdx.x - 1];

  bar();
}

int main (void) {
  unsigned int i;
  int *dg, hg[N];
  int sum = 0;

  foo<<<1, N>>>();

  if(cudaGetSymbolAddress((void**)&dg, g)){
      printf("couldn't get the symbol addr\n");
      return 1;
  }
  if(cudaMemcpy(hg, dg, N * sizeof(int), cudaMemcpyDeviceToHost)){
      printf("couldn't memcpy\n");
      return 1;
  }

  for (i = 0; i < N; i++) {
    sum += hg[i];
  }
  // 1+2+3+...+7+8=36
  if (sum == 36) {
    printf("PASSED\n");
  } else {
    printf("FAILED (%d)\n", sum);
  }

  return 0;
}
```

下图基本上描述了上述代码的编译过程：

<img src="/2019/03/04/CUDA学习系列(1)%20%7C%20编译链接篇/nvcc-whole-compilation.png" width="45%" style="display: block; margin: 0 auto;">

对应的编译指令如下，分别对应上图的三个阶段

```bash
# device code指的是CUDA相关代码，host object指的是c++代码编译出来的产物
# 将x.cu和y.cu中的device code分别嵌入到其对应的host object中，即x.o和y.o
nvcc --gpu-architecture=sm_50 --device-c x.cu y.cu
# 使用device-link将x.o和y.o中的device code链接在一起，得到link.o
nvcc --gpu-architecture=sm_50 --device-link x.o y.o --output-file a_dlink.o
# 将链接后的link.o和其他host object链接在一起，得到最终产物
# 这里<path>替换成你libcudart.so对应路径
g++ x.o y.o a_dlink.o -L<path> -lcudart
```

简单点说就是先将所有device code链接到一起，再和其它host object链接成最终产物。至此，对CUDA编译的整体流程上有了大概的了解。


回头再看上面三行编译指令，nvcc、–device-link这些命令或参数就比较容易理解了，但是还不够，比如上述指令中的–gpu-architecture=sm_50和–device-c是什么意思呢？这就涉及到CUDA编译的另外两个重要概念：**1.Separate Compilation** **2.Virtual and Real Architectures**。

## 0x02 : Separate Compilation

CUDA程序的编译，尤其是大型的CUDA程序编译过程中，Separate Compilation起到了举足轻重作用。要说明白Separate Compilation是什么，就得知道为什么要有Separate Compilation这个机制？


CUDA 5.0之前，是无法支持extern和static这两个关键词的(这部分我没有考究，欢迎大家指正)，这就意味着对于0x01章节中的那段代码，是无法编译的，因为你无法跨文件访问device变量和函数。对于CUDA这种经常会使用全局变量和全局函数的程序，支持extern和static这两个关键词就显得至关重要。extern可以让你跨文件访问device变量和函数，static保证了同名的device变量在不同文件内符号是不同的。所以在此之前，要编译CUDA程序，必须要将所有的device变量和函数放到一个文件里，这也被称作Whole Program Compilation。


对于0x01章节中那个代码，如果使用Separate Compilation，那么device相关编译将会分为两步。第一步是将**relocatable device code**编译到对应host object中，比如x.o和y.o。第二步是使用nvlink将x.o和y.o中的device code链接到一起得到a_dlink.o。这里之所以称第一步编译的device code为relocatable，意思是说这些device code在host object的位置会在第二步重新定位(relocatable)。对比Whole Program Compilation，我们称其device code为executable device code，意思是编译后的device code在host object中已经定位好了，一直到生成可执行文件都是不需要重新定位的(executable)。


当然，CUDA 5.0以后虽然支持Separate Compilation了，但是默认也是不开启的，比如下面我直接使用–compile进行编译，就会报错。

```bash
# y.h(4): warning: extern declaration of the entity g is treated as a static definition
# ptxas fatal   : Unresolved extern function '_Z3barv'
nvcc --gpu-architecture=sm_50 --compile x.cu y.cu               
```

下图对Separate Compilation的流程做了简单描述：

<img src="/2019/03/04/CUDA学习系列(1)%20%7C%20编译链接篇/nvcc-options-for-separate-compilation.png" width="65%" style="display: block; margin: 0 auto;">

## 0x03 : Virtual and Real Architectures

大家如果看过CUDA的编译选项，一定记得看过类似下面这条指令：

```bash
nvcc x.cu --gpu-architecture=compute_50 --gpu-code=sm_50
```

尤其是对其中compute_xx和sm_xx感到困惑。其实compute_xx就是对应GPU的Virtual Architecture，而sm_xx就是对应GPU的Real Architecture。至于后面的数字，代表的是GPU不同架构的版本。


不同的架构其实对应的是指令集的不同，类比CPU指令中的x86、amd64等。所以这里的Virtual和Real架构其实也是两种不同的指令集，其中Virtual Architecture会生成一种中间产物PTX(Parallel Thread Execution)，可以认为它是Virtual Architecture的汇编产物。Virtual Architecture是一个通用的指令集，主要是为了考虑不同显卡之间的兼容性。Real Architecture提供的是真实GPU上的指令集，也是最终CUDA程序运行的指令集。所以一般在选择编译选项的时候，Virtual Architecture的版本要选择低一些，因为这样可以大大提高兼容性，也就是说可以跑在更多的CUDA机器上。而Real Architecture尽量使用最新的版本，因为一般来说最新的版本会进行更多的优化。


从下图我们可以看出，CUDA程序编译时，首先会根据你指定的Virtual Architecture选项生成.ptx文件，然后再根据Real Architecture的选项将.ptx文件编译成.cubin文件，最终再经过一系列处理.cubin文件会链接到目标产物中。

<img src="/2019/03/04/CUDA学习系列(1)%20%7C%20编译链接篇/virtual-architectures.png" width="60%" style="display: block; margin: 0 auto;">

因为每一代的显卡的指令集是不一样的，而你在编译CUDA程序的时候，是不知道你的程序最终会跑在哪种显卡上，也就是说你不知道应该指定sm_xx什么版本。所以就出现了Just-in-Time Compilation的概念

## 0x04 : Just-in-Time Compilation

JIT的概念使用的比较广泛，比如Java就使用了JIT的概念，可以将.ptx文件类比Java的.class文件。JIT的好处太多了，这里就不赘述了。有兴趣的可以参考知乎上的讨论[如何通俗易懂地介绍「即时编译」（JIT），它的优点和缺点是什么？](https://www.zhihu.com/question/21093419)。


简单点说，CUDA中的JIT就是在CUDA程序运行时，将.ptx文件根据目标平台编译为对应的.cubin文件，并链接到目标产物中。当然，这会造成程序启动会慢一些，因为要根据平台生成二进制文件，但是也只是第一次启动比较慢，因为之后通常会将刚刚生成的二进制文件缓存下来。


下图阐述的就是CUDA的JIT机制。

<img src="/2019/03/04/CUDA学习系列(1)%20%7C%20编译链接篇/just-in-time-compilation.png" width="60%" style="display: block; margin: 0 auto;">

## 0x05 : 整体流程详解

CUDA的整体编译链接过程如图所示：

<img src="/2019/03/04/CUDA学习系列(1)%20%7C%20编译链接篇/cuda-compilation-from-cu-to-executable.png" width="80%" style="display: block; margin: 0 auto;">

上面这幅流程图基本上将CUDA的编译链接过程描绘得比较详细，接下来我再结合一个实际的CUDA编译链接案例对这幅流程图进行详细说明。


为了方便，我选择了cuda samples里面的vectorAdd程序进行编译。当然我需要在编译选项中添加-dryrun选项，保证可以在控制台上列出所有的编译子命令而不进行真实的编译。下面就是控制台打印出来的vectorAdd.cu编译的整个过程。

```bash
vectorAdd nvcc -dryrun --include-path="../../common/inc" vectorAdd.cu
# $ _SPACE_=
# $ _CUDART_=cudart
# $ _HERE_=/usr/local/cuda/bin
# $ _THERE_=/usr/local/cuda/bin
# $ _TARGET_SIZE_=
# $ _TARGET_DIR_=
# $ _TARGET_DIR_=targets/x86_64-linux
# $ TOP=/usr/local/cuda/bin/..
# $ NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
# $ LD_LIBRARY_PATH=/usr/local/cuda/bin/../lib:"":/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64":/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
# $ PATH=/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/home/pjx/anaconda3/bin:/usr/local/cuda/bin:/usr/local/MATLAB/R2017b/bin:/home/pjx/anaconda3/bin:/usr/local/cuda/bin:/home/pjx/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/lib/jvm/java-11-oracle/bin:/usr/lib/jvm/java-11-oracle/db/bin:/home/pjx/bin
# $ INCLUDES="-I/usr/local/cuda/bin/../targets/x86_64-linux/include"
# $ LIBRARIES=  "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"
# $ CUDAFE_FLAGS=
# $ PTXAS_FLAGS=
# $ gcc -std=c++14 -D__CUDA_ARCH__=300 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  -I"../../common/inc" "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=0 -D__CUDACC_VER_BUILD__=130 -include "cuda_runtime.h" -m64 "vectorAdd.cu" > "/tmp/tmpxft_000003b0_00000000-8_vectorAdd.cpp1.ii"
# $ cicc --c++14 --gnu_version=70300 --allow_managed   -arch compute_30 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_000003b0_00000000-2_vectorAdd.fatbin.c" -tused -nvvmir-library "/usr/local/cuda/bin/../nvvm/libdevice/libdevice.10.bc" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_000003b0_00000000-3_vectorAdd.module_id" --orig_src_file_name "vectorAdd.cu" --gen_c_file_name "/tmp/tmpxft_000003b0_00000000-5_vectorAdd.cudafe1.c" --stub_file_name "/tmp/tmpxft_000003b0_00000000-5_vectorAdd.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_000003b0_00000000-5_vectorAdd.cudafe1.gpu"  "/tmp/tmpxft_000003b0_00000000-8_vectorAdd.cpp1.ii" -o "/tmp/tmpxft_000003b0_00000000-5_vectorAdd.ptx"
# $ ptxas -arch=sm_30 -m64  "/tmp/tmpxft_000003b0_00000000-5_vectorAdd.ptx"  -o "/tmp/tmpxft_000003b0_00000000-9_vectorAdd.sm_30.cubin"
# $ fatbinary --create="/tmp/tmpxft_000003b0_00000000-2_vectorAdd.fatbin" -64 "--image=profile=sm_30,file=/tmp/tmpxft_000003b0_00000000-9_vectorAdd.sm_30.cubin" "--image=profile=compute_30,file=/tmp/tmpxft_000003b0_00000000-5_vectorAdd.ptx" --embedded-fatbin="/tmp/tmpxft_000003b0_00000000-2_vectorAdd.fatbin.c" --cuda
# $ rm /tmp/tmpxft_000003b0_00000000-2_vectorAdd.fatbin
# $ gcc -std=c++14 -E -x c++ -D__CUDACC__ -D__NVCC__  -I"../../common/inc" "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=0 -D__CUDACC_VER_BUILD__=130 -include "cuda_runtime.h" -m64 "vectorAdd.cu" > "/tmp/tmpxft_000003b0_00000000-4_vectorAdd.cpp4.ii"
# $ cudafe++ --c++14 --gnu_version=70300 --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_000003b0_00000000-5_vectorAdd.cudafe1.cpp" --stub_file_name "tmpxft_000003b0_00000000-5_vectorAdd.cudafe1.stub.c" --module_id_file_name "/tmp/tmpxft_000003b0_00000000-3_vectorAdd.module_id" "/tmp/tmpxft_000003b0_00000000-4_vectorAdd.cpp4.ii"
# $ gcc -std=c++14 -D__CUDA_ARCH__=300 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -I"../../common/inc" "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 -o "/tmp/tmpxft_000003b0_00000000-10_vectorAdd.o" "/tmp/tmpxft_000003b0_00000000-5_vectorAdd.cudafe1.cpp"
# $ nvlink --arch=sm_30 --register-link-binaries="/tmp/tmpxft_000003b0_00000000-6_a_dlink.reg.c"  -m64   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_000003b0_00000000-10_vectorAdd.o"  -o "/tmp/tmpxft_000003b0_00000000-11_a_dlink.sm_30.cubin"
# $ fatbinary --create="/tmp/tmpxft_000003b0_00000000-7_a_dlink.fatbin" -64 -link "--image=profile=sm_30,file=/tmp/tmpxft_000003b0_00000000-11_a_dlink.sm_30.cubin" --embedded-fatbin="/tmp/tmpxft_000003b0_00000000-7_a_dlink.fatbin.c"
# $ rm /tmp/tmpxft_000003b0_00000000-7_a_dlink.fatbin
# $ gcc -std=c++14 -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_000003b0_00000000-7_a_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_000003b0_00000000-6_a_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  -I"../../common/inc" "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=0 -D__CUDACC_VER_BUILD__=130 -m64 -o "/tmp/tmpxft_000003b0_00000000-12_a_dlink.o" "/usr/local/cuda/bin/crt/link.stub"
# $ g++ -m64 -o "a.out" -std=c++14 -Wl,--start-group "/tmp/tmpxft_000003b0_00000000-12_a_dlink.o" "/tmp/tmpxft_000003b0_00000000-10_vectorAdd.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group
```

上面这部分控制台的打印内容还是相当多的，这里我对其进行了简化，并结合上述的编译流程图，方便大家看出里面最核心的内容。

```bash
vectorAdd nvcc -dryrun --include-path="../../common/inc" vectorAdd.cu
# $ gcc "vectorAdd.cu" > "8_vectorAdd.cpp1.ii"
# $ cicc -arch compute_30 --include_file_name "2_vectorAdd.fatbin.c" --orig_src_file_name "vectorAdd.cu" --gen_c_file_name "5_vectorAdd.cudafe1.c" --stub_file_name "5_vectorAdd.cudafe1.stub.c" "8_vectorAdd.cpp1.ii" -o "5_vectorAdd.ptx"
# $ ptxas -arch=sm_30 "5_vectorAdd.ptx" -o "9_vectorAdd.sm_30.cubin"
# $ fatbinary --create="2_vectorAdd.fatbin" "--image=profile=sm_30,file=9_vectorAdd.sm_30.cubin" "--image=profile=compute_30,file=5_vectorAdd.ptx" --embedded-fatbin="2_vectorAdd.fatbin.c"
# $ gcc "vectorAdd.cu" > "4_vectorAdd.cpp4.ii"
# $ cudafe++ --gen_c_file_name "5_vectorAdd.cudafe1.cpp" --stub_file_name "5_vectorAdd.cudafe1.stub.c" "4_vectorAdd.cpp4.ii"
# $ gcc -o "10_vectorAdd.o" "5_vectorAdd.cudafe1.cpp"
# $ nvlink --register-link-binaries="6_a_dlink.reg.c" "10_vectorAdd.o" -o "11_a_dlink.sm_30.cubin"
# $ fatbinary --create="7_a_dlink.fatbin" -link "--image=profile=sm_30,file=11_a_dlink.sm_30.cubin" --embedded-fatbin="7_a_dlink.fatbin.c"
# $ gcc -DFATBINFILE="\"7_a_dlink.fatbin.c\"" DREGISTERLINKBINARYFILE="\"6_a_dlink.reg.c\"" -o "12_a_dlink.o" "/usr/local/cuda/bin/crt/link.stub"
# $ g++ -o "a.out" --start-group "12_a_dlink.o" "10_vectorAdd.o"
```

1. 用gcc编译器对vectorAdd.cu文件进行预处理，生成8_vectorAdd.cpp1.ii文件，注意后缀名为ii的文件就是预处理后的C++文件。**这里有一个令人比较困惑的地方是，我明明调用的是nvcc，为什么会出来gcc指令？实际上nvcc只是一个编译器的调用集合，它会去调用很多其他的编译工具，比如gcc和后续的cicc、ptxas、fatbinary等等**。


2. 用cicc将8_vectorAdd.cpp1.ii文件处理为5_vectorAdd.ptx文件，也就是CUDA的Virtual Architecture的汇编文件。cicc是一个基于LLVM的高层优化器和PTX生成器。详细信息可以参考[Building GPU Compilers with libNVVM](http://on-demand.gputechconf.com/gtc/2013/presentations/S3185-Building-GPU-Compilers-libNVVM.pdf)。


3. 用ptxas将5_vectorAdd.ptx文件根据Real Architecture编译为9_vectorAdd.sm_30.cubin文件。注意对于指定的每一个Virtual Architecture，需要重复进行1\~3步。也就是整体编译链接图中的实线绿框部分。


4. 用fatbinary将不同的Virtual Architecture生成的.ptx文件和.cubin合并在一起生成.fatbin.c文件。之所以这里要叫fatbinary，其中fat表示的就是将.ptx和不同版本的.cubin文件一起塞到.fatbin.c中(.ptx代表Virtual Architecture，而.cubin代表Real Architecture)，当然这只是我的猜测。


5. 有了fatbin.c文件，CUDA部分的编译就暂时告一段落。接下来用gcc对vectorAdd.cu文件再进行一次预处理，得到4_vectorAdd.cpp4.ii文件。这次预处理主要是为了进行host部分的编译。


6. 用cudafe++将4_vectorAdd.cpp4.ii文件中的host和device部分进行分离，得到host部分5_vectorAdd.cudafe1.cpp文件。


7. 将分离的host部分代码5_vectorAdd.cudafe1.cpp结合刚才1\~4步编译的CUDA产物2_vectorAdd.fatbin.c编译为10_vectorAdd.o。注意5_vectorAdd.cudafe1.cpp包含了5_vectorAdd.cudafe1.stub.c，而5_vectorAdd.cudafe1.stub.c包含了2_vectorAdd.fatbin.c，这样就保证10_vectorAdd.o既有host部分，又有device部分。另外对于每一个.cu文件，循环执行1\~7步，对应整体编译链接图中的虚线绿框部分。整体来说，1\~7步其实是将.cu中的device部分交由右边流程处理(CUDA专用Compiler)，将host部分交由左边流程处理(CPP/C专用Compiler)，最终再将它们合并到一个object文件中。


8. 1\~7步产生了不同.cu文件对应产物.o文件。此时，进入了Separate Compilation的重要一环，就是将不同.o文件中的device code重新定位到同一个文件中，即使用nvlink将所有device code编译到11_a_dlink.sm_30.cubin文件中。


9. 有了统一之后的.cubin文件，再使用fatbinary将.cubin文件处理为7_a_dlink.fatbin.c文件，方便C Compiler进行统一编译。


10. 将7_a_dlink.fatbin.c文件结合6_a_dlink.reg.c和cuda下的link.stub生成device code对应的最终编译产物12_a_dlink.o。注意link.stub是cuda/bin/crt下的，而此处的crt表示CUDA C Runtime，至于link.stub本质是什么，我也不是很清楚。


11. 最后使用g++将10_vectorAdd.o和12_a_dlink.o链接为最终的目标产物a.out。也就是将host object和device object链接在一起。

## 0x06 : 心得体会

整篇文章比较简单，只是对CUDA的编译链接做了简单的阐述，很多话题没有更深入讨论，也是因为自己本身用到这部分的知识比较少，经验不足。之前是因为要在Qt Creator上使用CUDA，也就是需要利用qmake编译CUDA程序，参考的文章比较少，这才对CUDA的编译链接过程稍微做了一些了解。所以有些内容如果阐述的不详细或者有错误，请大家批评指正。

## 0x07 : 参考文献

- [cuda-compiler-driver-nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [Building GPU Compilers with libNVVM](http://on-demand.gputechconf.com/gtc/2013/presentations/S3185-Building-GPU-Compilers-libNVVM.pdf)
