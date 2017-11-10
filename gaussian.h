//
// Created by Xin Xu on 11/9/17.
//

#ifndef PARAPANO_GAUSSIAN_H
#define PARAPANO_GAUSSIAN_H

float* createGaussianFilter(int h, int w, float sigma);
void printGaussianFilter(float* filter, int h, int w);
float* applyGaussianFilter(float* img, float* filter, int h, int w, int fh, int fw);

#endif //PARAPANO_GAUSSIAN_H
