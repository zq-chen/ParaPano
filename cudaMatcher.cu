#include <string>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaMatcher.h"

#define NUM_THREADS_PER_BLOCK 1024

using namespace std;

CudaMatcher::CudaMatcher() {
    // num_matches = 0;
    cudaDesc1 = NULL;
    cudaDesc2 = NULL;
}

CudaMatcher::~CudaMatcher() {
    if (cudaDesc1) {
        // free image data on CUDA
        cudaFree(cudaDesc1);
        cudaFree(cudaDesc2);
    }
}

void copyDescriptorToDevice(vector<Descriptor>& desc, Descriptor* cudaDesc) {
    int num_desc = desc.size();
    printf("copyDescriptorToDevice start\n");

    // convert vector to array to be used on Cuda Device
    Descriptor* tempDesc = new Descriptor[num_desc];
    for (int i = 0; i < num_desc; i++) {
        tempDesc[i] = desc[i];
    }

    cudaMalloc((void**)&cudaDesc, sizeof(Descriptor) * num_desc);
    cudaMemcpy(cudaDesc, tempDesc, sizeof(Descriptor) * num_desc, cudaMemcpyHostToDevice);

    delete[] tempDesc;
    printf("copyDescriptorToDevice end\n");
}


void
CudaMatcher::setup(vector<Descriptor>& desc1, vector<Descriptor>& desc2) {
    // printCudaInfo();
    num_desc1 = desc1.size();
    num_desc2 = desc2.size();
    copyDescriptorToDevice(desc1, cudaDesc1);
    copyDescriptorToDevice(desc2, cudaDesc2);
}


__device__ __inline__ int countOneBits(int64_t i) {
    i = i - ((i >> 1) & 0x5555555555555555);
    i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
}


__device__ __inline__ int hammingDistance(Descriptor& d1, Descriptor& d2) {
    int dist = 0;
    dist += countOneBits(d1.num0 ^ d2.num0);
    dist += countOneBits(d1.num1 ^ d2.num1);
    dist += countOneBits(d1.num2 ^ d2.num2);
    dist += countOneBits(d1.num3 ^ d2.num3);
    return dist;
}

__device__ void findBestMatch(int idx, Descriptor& d, Descriptor* cudaDesc2, 
    int num_desc2, float* ratios, int* match_indices) {

    int min = INT_MAX;
    int second_min = INT_MAX;
    int min_idx = -1;
    for (int i = 0; i < num_desc2; i++) {
        int dist = hammingDistance(cudaDesc2[i], d);
        if (dist < min) {
            second_min = min;
            min = dist;
            min_idx = i;
        } else if (dist < second_min) {
            second_min = dist;
        }
    }
    float r = second_min == 0 ? 1 : float(min) / second_min;
    match_indices[idx] = min_idx;
    ratios[idx] = r;
}

__global__ void match(Descriptor* cudaDesc1, Descriptor* cudaDesc2, 
    int num_desc1, int num_desc2, float* ratios, int* match_indices) {

    int begin = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;
    int end = min(begin + NUM_THREADS_PER_BLOCK, num_desc1);

    for (int i = begin; i < end; i++) {
        findBestMatch(i, cudaDesc1[i], cudaDesc2, num_desc2, 
            ratios, match_indices);
    }
}

void
CudaMatcher::getMatchResult(float* ratios, int* match_indices,
                            float* cuda_ratios, int* cuda_match_indices) {

    cudaMemcpy(ratios, cuda_ratios, sizeof(float) * num_desc1,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(match_indices, cuda_match_indices, sizeof(int) * num_desc1,
        cudaMemcpyDeviceToHost);
}

void
CudaMatcher::findMatch() {
    float* ratios;
    float* cuda_ratios;
    int* match_indices;
    int* cuda_match_indices;
    float ratio_threshold = 0.8;

    ratios = new float[num_desc1];
    match_indices = new int[num_desc1];

    cudaMalloc(&cuda_ratios, sizeof(float) * num_desc1);
    cudaMalloc(&cuda_match_indices, sizeof(int) * num_desc1);
    printf("findMatch: malloc done\n");

    // Spawn CUDA threads
    int num_blocks = (num_desc1 + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    match<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(cudaDesc1, cudaDesc2, 
        num_desc1, num_desc2, cuda_ratios, cuda_match_indices);
    cudaDeviceSynchronize();
    printf("synch done\n");

    getMatchResult(ratios, match_indices, cuda_ratios, cuda_match_indices);

    for (int i = 0; i < num_desc1; i++) {
        if (ratios[i] < ratio_threshold) {
            indices1.push_back(i);
            indices2.push_back(match_indices[i]);
        }
    }
    printf("push back done\n");
    cudaFree(cuda_ratios);
    cudaFree(cuda_match_indices);
    delete[] ratios;
    delete[] match_indices;
}