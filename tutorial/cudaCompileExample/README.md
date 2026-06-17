# CUDA 从零开始：让 GPU 算 \(2+3\)

你现在先不要管 3DGS，也不要管复杂 GPU 优化。我们只做一件最小的事：

$$
c = a + b
$$

具体一点：

$$
2 + 3 = 5
$$

目标：写一个 CUDA 程序，让 GPU 帮我们算出 `5`。

对应文件是 [00_add_two_numbers.cu](/home/leo/Projects/SJTU/cudaCompile/00_add_two_numbers.cu)。

## 第 0 步：你要创建什么文件

进入项目目录：

```bash
cd /home/leo/Projects/SJTU/cudaCompile
```

创建这个文件：

```bash
touch 00_add_two_numbers.cu
```

这个文件要干嘛？

```text
00_add_two_numbers.cu
  用 CUDA C++ 写一个程序
  CPU 准备 2 和 3
  GPU 计算 2 + 3
  CPU 拿回结果并打印
```

为什么文件名是 `.cu`？

```text
.cpp 是普通 C++ 源文件
.cu 是 CUDA C++ 源文件
```

`.cu` 文件可以写普通 C++，也可以写 GPU kernel。

## 第 1 步：先写头文件

文件开头写：

```cpp
#include <cstdio>
#include <cstdlib>
```

意思：

```text
cstdio
  用来调用 std::printf，负责打印结果。

cstdlib
  用来调用 std::exit，出错时直接退出程序。
```

## 第 2 步：写一个检查 CUDA 错误的工具

写：

```cpp
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)
```

你现在只需要理解它的用途：

```text
每次调用 CUDA API 后检查有没有失败。
如果失败，就打印错误位置和错误原因。
```

例如后面我们会写：

```cpp
CHECK_CUDA(cudaMalloc(&d_a, sizeof(int)));
```

意思是：

```text
请 GPU 分配一块 int 大小的内存。
如果失败，马上打印错误并退出。
```

## 第 3 步：写 GPU 要执行的函数

写：

```cpp
__global__ void add_two_numbers_kernel(const int* a, const int* b, int* c) {
    *c = *a + *b;
}
```

逐个解释：

```text
__global__
  这表示这个函数是 GPU kernel。
  它由 CPU 启动，但在 GPU 上执行。

void
  这个函数不通过 return 返回结果。
  结果会写到 c 指向的 GPU 内存里。

add_two_numbers_kernel
  函数名，意思是“两个数字相加的 GPU kernel”。

const int* a
  a 是一个指针，指向 GPU 内存里的第一个整数。
  const 表示 kernel 不会修改 a。

const int* b
  b 是一个指针，指向 GPU 内存里的第二个整数。
  const 表示 kernel 不会修改 b。

int* c
  c 是一个指针，指向 GPU 内存里的结果。
  GPU 会把计算结果写到这里。

*c = *a + *b;
  取出 a 指向的整数。
  取出 b 指向的整数。
  相加。
  写进 c 指向的位置。
```

这里的 `*a`、`*b`、`*c` 可以先理解为：

```text
*a 表示 a 那块内存里存的值
*b 表示 b 那块内存里存的值
*c 表示 c 那块内存里存的值
```

## 第 4 步：写 CPU 主函数

程序从这里开始运行：

```cpp
int main() {
```

先在 CPU 内存里准备三个整数：

```cpp
int h_a = 2;
int h_b = 3;
int h_c = 0;
```

变量解释：

```text
h_a
  host a，也就是 CPU 内存里的 a。
  值是 2。

h_b
  host b，也就是 CPU 内存里的 b。
  值是 3。

h_c
  host c，也就是 CPU 内存里的结果。
  初始是 0，等 GPU 算完以后会变成 5。
```

这里的 `h_` 是命名习惯：

```text
h = host = CPU
```

## 第 5 步：准备 GPU 内存指针

写：

```cpp
int* d_a = nullptr;
int* d_b = nullptr;
int* d_c = nullptr;
```

变量解释：

```text
d_a
  device a，也就是 GPU 内存里的 a。

d_b
  device b，也就是 GPU 内存里的 b。

d_c
  device c，也就是 GPU 内存里的结果。

nullptr
  现在还没有真正分配 GPU 内存，所以先设为空。
```

这里的 `d_` 是命名习惯：

```text
d = device = GPU
```

## 第 6 步：在 GPU 上分配内存

写：

```cpp
CHECK_CUDA(cudaMalloc(&d_a, sizeof(int)));
CHECK_CUDA(cudaMalloc(&d_b, sizeof(int)));
CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
```

意思：

```text
cudaMalloc
  在 GPU 上分配内存。

&d_a
  把 d_a 这个指针的地址交给 cudaMalloc。
  cudaMalloc 会修改 d_a，让它指向真正的 GPU 内存。

sizeof(int)
  只需要放一个 int，所以分配一个 int 大小。
```

现在内存状态是：

```text
CPU:
  h_a = 2
  h_b = 3
  h_c = 0

GPU:
  d_a = 已分配，但还没有值
  d_b = 已分配，但还没有值
  d_c = 已分配，但还没有值
```

## 第 7 步：把 CPU 数据拷到 GPU

写：

```cpp
CHECK_CUDA(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));
```

逐个解释：

```text
cudaMemcpy
  拷贝内存。

d_a
  目标地址：GPU 上的 d_a。

&h_a
  源地址：CPU 上 h_a 的地址。

sizeof(int)
  拷贝一个 int 大小。

cudaMemcpyHostToDevice
  拷贝方向：CPU -> GPU。
```

现在内存状态是：

```text
CPU:
  h_a = 2
  h_b = 3
  h_c = 0

GPU:
  d_a = 2
  d_b = 3
  d_c = 未知
```

## 第 8 步：启动 GPU 计算

写：

```cpp
add_two_numbers_kernel<<<1, 1>>>(d_a, d_b, d_c);
```

解释：

```text
add_two_numbers_kernel
  我们刚才写的 GPU 函数。

<<<1, 1>>>
  启动配置。
  第一个 1：启动 1 个 block。
  第二个 1：每个 block 里启动 1 个 thread。

(d_a, d_b, d_c)
  把 GPU 内存地址传给 GPU kernel。
```

为什么这里用 `<<<1, 1>>>`？

```text
因为我们只算一个加法：2 + 3。
一个 GPU thread 就够了。
```

GPU 执行这句：

```cpp
*c = *a + *b;
```

也就是：

```text
d_c = d_a + d_b = 2 + 3 = 5
```

## 第 9 步：检查 kernel 有没有启动失败

写：

```cpp
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

解释：

```text
cudaGetLastError()
  检查刚才启动 kernel 有没有配置错误。

cudaDeviceSynchronize()
  等 GPU 计算完成。
  CPU 启动 kernel 后不会自动等待，所以这里必须等一下。
```

## 第 10 步：把结果从 GPU 拷回 CPU

写：

```cpp
CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
```

解释：

```text
&h_c
  目标地址：CPU 上 h_c 的地址。

d_c
  源地址：GPU 上的结果。

cudaMemcpyDeviceToHost
  拷贝方向：GPU -> CPU。
```

现在内存状态是：

```text
CPU:
  h_a = 2
  h_b = 3
  h_c = 5

GPU:
  d_a = 2
  d_b = 3
  d_c = 5
```

## 第 11 步：释放 GPU 内存

写：

```cpp
CHECK_CUDA(cudaFree(d_a));
CHECK_CUDA(cudaFree(d_b));
CHECK_CUDA(cudaFree(d_c));
```

解释：

```text
cudaFree
  释放之前 cudaMalloc 分配的 GPU 内存。
```

原则：

```text
有 cudaMalloc，就要有 cudaFree。
```

## 第 12 步：打印结果

写：

```cpp
std::printf("%d + %d = %d\n", h_a, h_b, h_c);
std::printf("check: %s\n", h_c == 5 ? "PASS" : "FAIL");

return h_c == 5 ? 0 : 1;
```

解释：

```text
第一行打印 2 + 3 = 5。

第二行检查 h_c 是否等于 5。
如果等于 5，打印 PASS。
否则打印 FAIL。

return 0 表示程序成功。
return 1 表示程序失败。
```

## 第 13 步：编译

编译命令：

```bash
cd /home/leo/Projects/SJTU/cudaCompile
nvcc -O2 -o add_two_numbers 00_add_two_numbers.cu
```

解释：

```text
nvcc
  NVIDIA CUDA 编译器。

-O2
  开启常规优化。

-o add_two_numbers
  输出文件名叫 add_two_numbers。

00_add_two_numbers.cu
  输入源代码文件。
```

编译成功后，会多出一个可执行文件：

```text
add_two_numbers
```

## 第 14 步：运行

运行：

```bash
./add_two_numbers
```

你应该看到：

```text
2 + 3 = 5
check: PASS
```

这说明：

```text
CPU 准备了 2 和 3
GPU 算出了 5
CPU 拿回结果并检查通过
```

## 第 15 步：用 Makefile 一键跑

这个目录还有 [Makefile](/home/leo/Projects/SJTU/cudaCompile/Makefile)。

你可以直接：

```bash
cd /home/leo/Projects/SJTU/cudaCompile
make clean
make
make run
```

`make run` 会运行：

```text
./add_two_numbers
./add_many_numbers
```

## 第 16 步：如果要算好几个怎么办

看 [01_add_many_numbers.cu](/home/leo/Projects/SJTU/cudaCompile/01_add_many_numbers.cu)。

现在我们不是只算一个：

$$
2 + 3 = 5
$$

而是算 8 个：

```text
c[0] = a[0] + b[0]
c[1] = a[1] + b[1]
c[2] = a[2] + b[2]
...
c[7] = a[7] + b[7]
```

输入是：

```cpp
int h_a[n] = {0, 1, 2, 3, 4, 5, 6, 7};
int h_b[n] = {100, 101, 102, 103, 104, 105, 106, 107};
int h_c[n] = {0, 0, 0, 0, 0, 0, 0, 0};
```

变量意思：

```text
n
  一共有几个数要算。这里 n = 8。

bytes
  这 8 个 int 一共占多少字节。
  cudaMalloc 和 cudaMemcpy 都需要知道字节数。

h_a
  CPU 内存里的第一个数组。

h_b
  CPU 内存里的第二个数组。

h_c
  CPU 内存里的结果数组。
```

GPU kernel 变成：

```cpp
__global__ void add_many_numbers_kernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

这里最重要的是：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

它给每个 GPU thread 算出一个编号。

这个编号就是它负责的数组位置：

```text
第 0 个 thread 算 c[0]
第 1 个 thread 算 c[1]
第 2 个 thread 算 c[2]
...
第 7 个 thread 算 c[7]
```

启动 GPU 时不再写 `<<<1, 1>>>`，而是：

```cpp
int threads_per_block = 4;
int blocks = (n + threads_per_block - 1) / threads_per_block;

add_many_numbers_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

这几个变量的意思：

```text
threads_per_block = 4
  每个 block 放 4 个 thread。

blocks
  需要几个 block。
  n = 8，threads_per_block = 4，所以 blocks = 2。

<<<blocks, threads_per_block>>>
  这里就是 <<<2, 4>>>。
  总共启动 2 * 4 = 8 个 GPU thread。
```

编译运行：

```bash
cd /home/leo/Projects/SJTU/cudaCompile
nvcc -O2 -o add_many_numbers 01_add_many_numbers.cu
./add_many_numbers
```

你应该看到：

```text
c[0] = 0 + 100 = 100
c[1] = 1 + 101 = 102
...
c[7] = 7 + 107 = 114
check: PASS
```
