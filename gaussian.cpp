//
// Created by Xin Xu on 11/9/17.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gaussian.h"

using namespace std;

// create a normalized gaussian filter of height h and width w
float* createGaussianFilter(const int h, const int w, float sigma) {
    float* gaussianFilter = new float[h*w];
    float sum = 0.0;
    int centerX = w/2;
    int centerY = h/2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int x = j - centerX;
            int y = i - centerY;
            float e = -(x*x+y*y)/(2*sigma*sigma);
            gaussianFilter[i * w + j] = exp(e)/(2*M_PI*sigma*sigma);
            sum += gaussianFilter[i * w + j];
        }
    }

    // normalize
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            gaussianFilter[i * w + j] /= sum;
        }
    }
    return gaussianFilter;
}

void printGaussianFilter(float* filter, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            cout<<filter[i * w + j]<<" ";
        }
        cout<<endl;
    }
}

inline bool inBound(int r, int c, int h, int w) {
    return r >= 0 && r < h && c >= 0 && c < w;
}

float* applyGaussianFilter(float* img, float* filter, int h, int w, int fh, int fw) {
    float* output = new float[h*w];
    int fhHalf = fh/2;
    int fwHalf = fw/2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float weightedSum = 0.0;
            for (int ii = -fhHalf; ii <= fhHalf; ii++) {
                for (int jj = -fwHalf; jj <= fwHalf; jj++) {
                    int r = i+ii;
                    int c = j+jj;
                    float imVal = inBound(r, c, h, w)? img[r*w+c] : 0;
                    weightedSum += imVal * filter[(ii+fhHalf)*fw + (jj+fwHalf)];
                }
            }
            output[i * w + j] = weightedSum;
        }
    }
    return output;
}