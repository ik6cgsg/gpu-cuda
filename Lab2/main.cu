#include <algorithm>
#include <new>
#include <cstddef>
#include <windows.h>
#include <iostream>
#include <cinttypes>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

static const int MIN = -30;
static const int MAX = 30;
static const unsigned int BLOCK_SIZE = 32;
static const size_t MATRIX_SIZES[3] = {1000, 2000, 3000};

enum class MultType { cpu, gpu, gpuShared };

__global__ void gpuKernMult(double* m1, double* m2, size_t size, double* res)
{
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size || j >= size) return;
    size_t ind = i * size + j;
    res[ind] = 0;
    for (size_t k = 0; k < size; k++) 
    {
        res[ind] += m1[i * size + k] * m2[k * size + j];
    }
}

__global__ void gpuKernMultShared(double* m1, double* m2, size_t size, double* res)
{
    size_t ty = threadIdx.y;
    size_t tx = threadIdx.x;
    size_t i = blockDim.y * blockIdx.y + ty;
    size_t j = blockDim.x * blockIdx.x + tx;
    double sum = 0;
    for (size_t ind = 0, aj = tx, bi = ty; ind * BLOCK_SIZE < size; ++ind, aj += BLOCK_SIZE, bi += BLOCK_SIZE)
    {
        __shared__ double a[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double b[BLOCK_SIZE][BLOCK_SIZE];
        a[ty][tx] = 0;
        b[ty][tx] = 0;
        if (i < size && aj < size)
        {
            a[ty][tx] = m1[i * size + aj];
        }
        if (j < size && bi < size)
        {
            b[ty][tx] = m2[bi * size + j];
        }
        __syncthreads();
        for (size_t k = 0; k < BLOCK_SIZE; k++)
        {
            sum += a[ty][k] * b[k][tx];
        }
        __syncthreads();
    }
    if (i < size && j < size)
    {
        res[i * size + j] = sum;
    }
}

static void initCudaMatr(double** m1, double** m2, double** res, size_t bytes, double* src1, double* src2)
{
    cudaMalloc(m1, bytes);
    cudaMalloc(m2, bytes);
    cudaMalloc(res, bytes);
    cudaMemcpy(*m1, src1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*m2, src2, bytes, cudaMemcpyHostToDevice);
}

static void initCudaTimer(cudaEvent_t* start, cudaEvent_t* end)
{
    cudaEventCreate(start);
    cudaEventCreate(end);
    cudaEventRecord(*start, 0);
}

static float countTime(cudaEvent_t start, cudaEvent_t end)
{
    float time;
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    return time;
}

static void destroyCudaObj(double* m1, double* m2, double* res, cudaEvent_t start, cudaEvent_t end)
{
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    cudaFree(res);
    cudaFree(m2);
    cudaFree(m1);
}

static float gpuMult(double* m1, double* m2, size_t size, double* res, MultType type)
{
    cudaEvent_t start, end;
    float time;
    double* cudaM1;
    double* cudaM2;
    double* cudaRes;
    size_t matrixBytesNum = sizeof(double) * size * size;
    dim3 cudaThreads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 cudaBlocks((size + cudaThreads.x - 1) / cudaThreads.x, (size + cudaThreads.y - 1) / cudaThreads.y);
    initCudaMatr(&cudaM1, &cudaM2, &cudaRes, matrixBytesNum, m1, m2);
    initCudaTimer(&start, &end);
    switch (type)
    {
    case MultType::gpu:
        gpuKernMult<<<cudaBlocks, cudaThreads>>>(cudaM1, cudaM2, size, cudaRes);
        break;
    case MultType::gpuShared:
        gpuKernMultShared<<<cudaBlocks, cudaThreads>>>(cudaM1, cudaM2, size, cudaRes);
        break;
    default:
        return -1;
    }
    time = countTime(start, end);
    cudaMemcpy(res, cudaRes, matrixBytesNum, cudaMemcpyDeviceToHost);
    destroyCudaObj(cudaM1, cudaM2, cudaRes, start, end);
    return time / 1000.0f;
}

static double deviation(double* m1, double* m2, size_t size)
{
    size_t n = size * size;
    double res = 0.0;
    for (size_t i = 0; i < n; i++) 
    {
        res = std::max(res, std::abs(m1[i] - m2[i]));
    }
    return res;
}

static double* randomMatrix(size_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(MIN, MAX);
    size_t n = size * size;
    double* res = new double[n];
    for (size_t i = 0; i < n; ++i) 
    {
        res[i] = distrib(gen);
    }
    return res;
}

static float cpuMult(double* m1, double* m2, size_t size, double* res)
{
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    for (size_t i = 0; i < size; i++) 
    {
        for (size_t j = 0; j < size; j++)
        {
            size_t ind = i * size + j;
            res[ind] = 0;
            for (size_t k = 0; k < size; k++) 
            {
                res[ind] += m1[i * size + k] * m2[k * size + j];
            }
        }
    }
    QueryPerformanceCounter(&end);
    return static_cast<float>(end.QuadPart - start.QuadPart) / freq.QuadPart;
}

static float mult(double* m1, double* m2, size_t size, double* res, MultType type)
{
    if (type == MultType::cpu)
    {
        return cpuMult(m1, m2, size, res);
    }
    else
    {
        return gpuMult(m1, m2, size, res, type);
    }
}

int main(int argc, char* argv[])
{
    for (size_t size: MATRIX_SIZES) 
    {
        std::cout << "Size == " << size << std::endl;
        double* m1 = randomMatrix(size);
        double* m2 = randomMatrix(size);
        double* gpuResMatr = new double[size * size];
        double* gpuSharedResMatr = new double[size * size];
        float gpuResTime = mult(m1, m2, size, gpuResMatr, MultType::gpu);
        float gpuSharedResTime = mult(m1, m2, size, gpuSharedResMatr, MultType::gpuShared);
        std::cout << "GPU result time == " << gpuResTime << std::endl;
        std::cout << "GPU + shared result time == " << gpuSharedResTime << std::endl;
        std::cout << "Maximum deviation: " << deviation(gpuResMatr, gpuSharedResMatr, size) << std::endl;
        delete[] gpuResMatr;
        delete[] gpuSharedResMatr;
        delete[] m2;
        delete[] m1;     
    }
    return 0;
}
