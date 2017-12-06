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

    // convert vector to array to be used on Cuda Device
    Descriptor* tempDesc = new Descriptor[num_desc];
    for (int i = 0; i < num_desc; i++) {
        tempDesc[i] = desc[i];
    }

    cudaMalloc((void**)&cudaDesc, sizeof(Descriptor) * num_desc);
    cudaMemcpy(cudaDesc, tempDesc, sizeof(Descriptor) * num_desc, cudaMemcpyHostToDevice);

    delete[] tempDesc;
}


void
CudaMatcher::setup(vector<Descriptor>& desc1, vector<Descriptor>& desc2) {
    // printCudaInfo();
    num_desc1 = desc1.size();
    num_desc2 = desc2.size();
    copyDescriptorToDevice(desc1, cudaDesc1);
    copyDescriptorToDevice(desc2, cudaDesc2);
}


__device__ __inline__ int countOneBits(int64_t x) {
    return 0;
}


__device__ __inline__ int hammingDistance(Descriptor& d1, Descriptor& d2) {
    int dist = 0;
    BitArray& b1 = d1.values;
    BitArray& b2 = d2.values;
    for (int i = 0; i < b1.num_cells; i++) {
        int64_t diff = b1.value[i] ^ b2.value[i];
        dist += countOneBits(diff);
    }
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
CudaMatcher::findMatch() {
    float* ratios;
    int* match_indices;
    float ratio_threshold = 0.8;

    cudaMalloc(&ratios, sizeof(float) * num_desc1);
    cudaMalloc(&match_indices, sizeof(int) * num_desc1);

    // Spawn CUDA threads
    int num_blocks = (num_desc1 + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    match<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(cudaDesc1, cudaDesc2, 
        num_desc1, num_desc2, ratios, match_indices);
    cudaDeviceSynchronize();

    for (int i = 0; i < num_desc1; i++) {
        if (ratios[i] < ratio_threshold) {
            indices1.push_back(i);
            indices2.push_back(match_indices[i]);
        }
    }
    cudaFree(ratios);
    cudaFree(match_indices);
}