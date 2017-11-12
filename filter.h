//
// Created by Xin Xu on 11/9/17.
//

#ifndef PARAPANO_GAUSSIAN_H
#define PARAPANO_GAUSSIAN_H

//float* createGaussianFilter(int h, int w, float sigma);
void printGaussianFilter(float* filter, int h, int w);
//float* applyFilter(float* img, float* filter, int h, int w, int fh, int fw);
//float** createGaussianFilters(int fsize, float sigma0, float k, int* levels, int num_levels);
float** createGaussianPyramid(float* img, int h, int w, float sigma0, float k, int* levels, int num_levels);

#endif //PARAPANO_GAUSSIAN_H
