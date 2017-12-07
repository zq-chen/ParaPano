#pragma once

#include "brief.h"

class CudaMatcher {
public:
    std::vector<int> indices1;
    std::vector<int> indices2;

    int num_desc1;
    int num_desc2;

    // data on cuda devices
    Descriptor* cudaDesc1;
    Descriptor* cudaDesc2;

	CudaMatcher();
    virtual ~CudaMatcher();

    void setup(std::vector<Descriptor>& desc1, std::vector<Descriptor>& desc2);
    void findMatch();

    void getMatchResult(float* ratios, int* match_indices,
                        float* cuda_ratios, int* cuda_match_indices);
};