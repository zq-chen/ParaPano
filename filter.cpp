//
// Created by Xin Xu on 11/9/17.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "filter.h"


// create a normalized gaussian filter of height h and width w
float* createGaussianFilter(const int fh, const int fw, float sigma) {
    float* gaussianFilter = new float[fh * fw];
    float sum = 0.0;
    int centerX = fw/2;
    int centerY = fh/2;
    for (int i = 0; i < fh; i++) {
        for (int j = 0; j < fw; j++) {
            int x = j - centerX;
            int y = i - centerY;
            float e = -(x*x + y*y) / (2 * sigma * sigma);
            gaussianFilter[i * fw + j] = exp(e) / (2 * M_PI * sigma * sigma);
            sum += gaussianFilter[i * fw + j];
        }
    }

    // normalize
    for (int i = 0; i < fh; i++) {
        for (int j = 0; j < fw; j++) {
            gaussianFilter[i * fw + j] /= sum;
        }
    }
    return gaussianFilter;
}

void printGaussianFilter(float* filter, int fh, int fw) {
    for (int i = 0; i < fh; i++) {
        for (int j = 0; j < fw; j++) {
            printf("%.4f ", filter[i * fw + j]);
        }
        std::cout << std::endl;
    }
}

inline bool inBound(int r, int c, int h, int w) {
    return r >= 0 && r < h && c >= 0 && c < w;
}

float* applyFilter(float* img, float* filter, int h, int w, int fh, int fw) {
    float* output = new float[h * w];
    int fhHalf = fh/2;
    int fwHalf = fw/2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float weightedSum = 0.0;
            for (int ii = -fhHalf; ii < fh - fhHalf; ii++) {
                for (int jj = -fwHalf; jj < fw - fwHalf; jj++) {
                    int r = i + ii;
                    int c = j + jj;
                    float imVal = inBound(r, c, h, w)? img[r * w + c] : 0;
                    weightedSum += imVal * filter[(ii+fhHalf)*fw + (jj+fwHalf)];
                }
            }
            output[i * w + j] = weightedSum;
        }
    }
    return output;
}

float** createGaussianPyramid(float* img, int h, int w, float sigma0, float k,
                              int* levels, int num_levels) {

    float** gaussian_pyramid = new float*[num_levels];
    for (int i = 0; i < num_levels; i++) {
        float sigma = sigma0 * pow(k, levels[i]);
        int fsize = floor(3 * sigma * 2) + 1;
        float* filter = createGaussianFilter(fsize, fsize, sigma);
        gaussian_pyramid[i] = applyFilter(img, filter, h, w, fsize, fsize);
        delete[] filter;
    }
    return gaussian_pyramid;
}