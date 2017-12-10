#include <string>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaMatcher.h"

#define NUM_THREADS_PER_BLOCK 512

using namespace std;

CudaMatcher::CudaMatcher() {
    cudaDesc1 = NULL;
    cudaDesc2 = NULL;
    cuda_ratios = NULL;
    cuda_match_indices = NULL;
}

CudaMatcher::~CudaMatcher() {
    // if (cudaDesc1) {
    //     // free image data on CUDA
    //     cudaFree(cudaDesc1);
    //     cudaFree(cudaDesc2);
    //     cudaFree(cuda_ratios);
    //     cudaFree(cuda_match_indices);
    // }
}

Descriptor* copyDescriptorToDevice(vector<Descriptor>& desc) {

    int num_desc = desc.size();

    // convert vector to array to be used on Cuda Device
    // Descriptor* tempDesc = (Descriptor*) malloc(sizeof(Descriptor) * num_desc);
    Descriptor* tempDesc = new Descriptor[num_desc];
    for (int i = 0; i < num_desc; i++) {
        tempDesc[i] = desc[i];
    }

    Descriptor* cudaDesc;
    cudaMalloc((void**)&cudaDesc, sizeof(Descriptor) * num_desc);
    cudaMemcpy(cudaDesc, tempDesc, sizeof(Descriptor) * num_desc, cudaMemcpyHostToDevice);

    return cudaDesc;
}

void
CudaMatcher::setup(vector<Descriptor> desc1, vector<Descriptor> desc2) {
    // printCudaInfo();
    num_desc1 = desc1.size();
    num_desc2 = desc2.size();
    cudaDesc1 = copyDescriptorToDevice(desc1);
    cudaDesc2 = copyDescriptorToDevice(desc2);
}


__device__ __inline__ int countOneBits(uint64_t i) {
    i = i - ((i >> 1) & 0x5555555555555555);
    i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
    // int count = 0;
    // while(i) {
    //   i &= (i - 1);
    //   count++;
    // }
    // return count;
}


__device__ __inline__ int hammingDistance(Descriptor d1, Descriptor d2) {
    int dist = 0;
    dist += countOneBits(d1.num0 ^ d2.num0);
    dist += countOneBits(d1.num1 ^ d2.num1);
    dist += countOneBits(d1.num2 ^ d2.num2);
    dist += countOneBits(d1.num3 ^ d2.num3);
    // for (int i = 0; i < 256; i++) { // XOR num0 to num3
    //     int temp = (d1[i] == d2[i])? 0 : 1;
    //     dist += temp;
    // }
    return dist;
}

__global__ void match(Descriptor* cudaDesc1, Descriptor* cudaDesc2,
    int num_desc1, int num_desc2, float* ratios, int* match_indices) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_desc1) {
        return;
    }

    // int descriptor_size = 256;
    // bool* d = (bool*)(&(cudaDesc1[idx * descriptor_size]));

    Descriptor d1 = cudaDesc1[idx];

    int min = INT_MAX;
    int second_min = INT_MAX;
    int min_idx = -1;
    for (int i = 0; i < num_desc2; i++) {
        // bool* desc2 = &(cudaDesc2[i * descriptor_size]);
        int dist = hammingDistance(d1, cudaDesc2[i]);
        if (dist < min) {
            second_min = min;
            min = dist;
            min_idx = i;
        } else if (dist < second_min) {
            second_min = dist;
        }
    }

    float r = second_min == 0 ? 1 : float(min) / second_min;
    // printf("threadIdx=%d, min_dist=%d\n", idx, min);
    match_indices[idx] = min_idx;
    ratios[idx] = r;
}

void
CudaMatcher::getMatchResult(float* ratios, int* match_indices) {

    cudaMemcpy(ratios, cuda_ratios, sizeof(float) * num_desc1,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(match_indices, cuda_match_indices, sizeof(int) * num_desc1,
        cudaMemcpyDeviceToHost);
}



MatchResult
CudaMatcher::findMatch() {
    float ratio_threshold = 0.8;
    float* ratios = (float*) malloc(sizeof(float) * num_desc1);
    int* match_indices = (int*) malloc(sizeof(int) * num_desc1);

    cudaError_t err = cudaMalloc(&cuda_ratios, sizeof(float) * num_desc1);
    err = cudaMalloc(&cuda_match_indices, sizeof(int) * num_desc1);
    if (err) throw err;

    // Spawn CUDA threads
    int num_blocks = (num_desc1 + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    printf("spwan %d threads in %d threaded blocks\n", num_desc1, num_blocks);

    match<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(cudaDesc1, cudaDesc2, 
        num_desc1, num_desc2, cuda_ratios, cuda_match_indices);
    cudaDeviceSynchronize();

    getMatchResult(ratios, match_indices);

    MatchResult match_result;
    for (int i = 0; i < num_desc1; i++) {
        if (ratios[i] < ratio_threshold) {
            match_result.indices1.push_back(i);
            match_result.indices2.push_back(match_indices[i]);
        }
    }

    free(ratios);
    free(match_indices);
    return match_result;
}
