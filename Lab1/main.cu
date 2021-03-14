#include <algorithm>
#include <new>
#include <cstddef>
#include <windows.h>
#include <iostream>
#include <cinttypes>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static const int MIN = -100;
static const int MAX =  100;
static const unsigned int BLOCK_SIZE = 32;
static const size_t MATRIX_SIZES[3] = {1000, 2000, 3000};

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

static float gpuMult(double* m1, double* m2, size_t size, double* res)
{
    cudaEvent_t start, end;
    float gpuTime = 0.0f;
    double* cudaM1;
    double* cudaM2;
    double* cudaRes;
    size_t matrixBytesNum = sizeof(double) * size * size;
    dim3 cudaThreads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 cudaBlocks((size + cudaThreads.x - 1) / cudaThreads.x, (size + cudaThreads.y - 1) / cudaThreads.y);
    cudaMalloc(reinterpret_cast<void**>(&cudaM1), matrixBytesNum);
    cudaMalloc(reinterpret_cast<void**>(&cudaM2), matrixBytesNum);
    cudaMalloc(reinterpret_cast<void**>(&cudaRes), matrixBytesNum);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    cudaMemcpy(cudaM1, m1, matrixBytesNum, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaM2, m2, matrixBytesNum, cudaMemcpyHostToDevice);
    gpuKernMult<<<cudaBlocks, cudaThreads>>>(cudaM1, cudaM2, size, cudaRes);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpuTime, start, end);
    cudaMemcpy(res, cudaRes, matrixBytesNum, cudaMemcpyDeviceToHost);
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    cudaFree(cudaM2);
    cudaFree(cudaRes);
    cudaFree(cudaM1);
    return gpuTime / 1000.0f;
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

int main(int argc, char* argv[])
{
    for (size_t size: MATRIX_SIZES) 
    {
        std::cout << "Size == " << size << std::endl;
        double* m1 = randomMatrix(size);
        double* m2 = randomMatrix(size);
        double* cpuResMatr = new double[size * size];
        double* gpuResMatr = new double[size * size];
        float cpuResTime = cpuMult(m1, m2, size, cpuResMatr);
        float gpuResTime = gpuMult(m1, m2, size, gpuResMatr);
        std::cout << "CPU result time == " << cpuResTime << std::endl;
        std::cout << "GPU result time == " << gpuResTime << std::endl;
        std::cout << "Maximum deviation: " << deviation(cpuResMatr, gpuResMatr, size) << std::endl;
        delete[] gpuResMatr;
        delete[] cpuResMatr;
        delete[] m2;
        delete[] m1;     
    }
    return 0;
}
