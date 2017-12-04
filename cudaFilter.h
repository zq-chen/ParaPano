#pragma once

class CudaFilterer {
public:
	float** gaussian_pyramid; // result on Host

	float* cudaImageData;	
	int imageWidth;
	int imageHeight;
	int numLevels;

	CudaFilterer();
    virtual ~CudaFilterer();

    void setup(float* img, int h, int w);

    float** createGaussianPyramid(float sigma0, float k, int* levels, int num_levels);
}