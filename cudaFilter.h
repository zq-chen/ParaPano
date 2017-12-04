#pragma once

class CudaFilterer {
public:
	float** gaussian_pyramid; // result on Host

	float* cudaImageData;
	float* cudaGaussianPyramid;	
	int imageWidth;
	int imageHeight;
	int numLevels;

	CudaFilterer();
    virtual ~CudaFilterer();

    void setup(float* img, int h, int w);

    void allocHostGaussianPyramid(int width, int height, int num_levels);
    void allocDeviceGaussianPyramid(int width, int height);

    float** createGaussianPyramid(float sigma0, float k, const int* levels, 
    							  int num_levels);

    // Get the ith Gaussian pyramid from device to host
    void getGaussianPyramid(int i);
};