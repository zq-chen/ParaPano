//
// Created by Xin Xu on 11/10/17.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "filter.h"
#include "keyPointDetector.h"

using namespace cv;

float** createDoGPyramid(float** gaussian_pyramid, int h, int w,
                         int num_levels) {

    float** dog_pyramid = new float*[num_levels-1];
    for (int k = 0; k < num_levels - 1; k++) {
        float* dog = new float[h*w];
        float* g0 = gaussian_pyramid[k];
        float* g1 = gaussian_pyramid[k + 1];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                dog[i * w + j] = g1[i * w + j] - g0[i * w + j];
            }
        }
        dog_pyramid[k] = dog;
    }
    return dog_pyramid;
}


inline bool inBound(int r, int c, int h, int w) {
    return r >= 0 && r < h && c >= 0 && c < w;
}

//  dx(i) = (I(i+1)-I(i-1))/2
float computeDx(float* img, int h, int w, int row, int col) {
    if (!inBound(row, col, h, w)) {
        return 0;
    }
    float left = col > 0? img[row * w + col - 1] : 0;
    float right = col < w - 1? img[row * w + col + 1] : 0;
    return (right - left) / 2.0;
}

float computeDy(float* img, int h, int w, int row, int col) {
    if (!inBound(row, col, h, w)) {
        return 0;
    }
    float up = row > 0? img[(row - 1)*w + col] : 0;
    float down = row < h - 1? img[(row + 1)*w + col] : 0;
    return  (up - down) / 2.0;
}

float computePrincipalCurvature(float* img, int h, int w, int row, int col) {

    // compute Hessian: H = [dxx dxy; dyx dyy]
    float left_dx = computeDx(img, h, w, row, col - 1);
    float right_dx = computeDx(img, h, w, row, col + 1);
    float dxx = (right_dx - left_dx) / 2.0;

    float up_dy = computeDy(img, h, w, row - 1, col);
    float down_dy = computeDy(img, h, w, row + 1, col);
    float dyy = (up_dy - down_dy) / 2.0;

    float up_dx = computeDx(img, h, w, row - 1, col);
    float down_dx = computeDx(img, h, w, row + 1, col);
    float dxy = (up_dx - down_dx) / 2.0;
    float dyx = dxy;

    // R = Tr(H)^2/Det(H)
    float det = dxx * dyy - dxy * dyx;
    float trace = dxx + dyy;
    return trace * trace / det;
}


void setLocalExtremaBoolean(float val, float temp_val, bool& isLocalMin,
                            bool& isLocalMax) {
    if (temp_val > val) {
        isLocalMax = false;
    } else if (temp_val < val) {
        isLocalMin = false;
    }
}

// check neighborhood with size patch_size * patch_size
bool isLocalExtrema(float* dog, float* dog_prev, float* dog_next, int h, int w,
                    int row, int col, int patch_size) {

    int psize = patch_size/2;
    float val = dog[row * w + col];
    bool isLocalMin = true;
    bool isLocalMax = true;
    for (int i = -psize; i <= psize; i++) {
        for (int j = -psize; j <= psize; j++) {
            int r = row + i;
            int c = col + j;
            if (inBound(r, c, h, w)) {
                setLocalExtremaBoolean(val, dog[r * w + c],
                                       isLocalMin, isLocalMax);
            }
        }
    }

    if (dog_prev != NULL) {
        setLocalExtremaBoolean(val, dog_prev[row * w + col],
                               isLocalMin, isLocalMax);
    }

    if (dog_next != NULL) {
        setLocalExtremaBoolean(val, dog_next[row * w + col],
                               isLocalMin, isLocalMax);
    }
    return isLocalMin || isLocalMax;
}

std::vector<Point> getLocalExtrema(float** dog_pyramid, int num_levels, int h,
                                   int w, float th_contrast, float th_r) {

    std::vector<Point> keypoints;
    int patch_size = 3;
    for (int i = 0; i < h; i++) { // row
        for (int j = 0; j < w; j++) { // col

            for (int k = 0; k < num_levels; k++) {

                float* dog = dog_pyramid[k];
                float* dog_prev = k > 0? dog_pyramid[k-1] : NULL;
                float* dog_next = k < num_levels - 1? dog_pyramid[k+1] : NULL;

                // remove point if the magnitude of DoG response is too small
                float val = dog[i * w + j];
                if (abs(val) <= th_contrast) {
                    continue;
                }

                // is local min or max
                bool is_local_extrema = isLocalExtrema(dog, dog_prev, dog_next,
                                                       h, w, i, j, patch_size);
                if (!is_local_extrema) {
                    continue;
                }

                // remove edge-like points - large principal curvature
                float principal_curvature = computePrincipalCurvature(dog, h, w,
                                                                      i, j);
                if (principal_curvature > th_r) {
                    continue;
                }

                // point is a key point
                keypoints.push_back(Point(j, i));
                break;

            }
        }
    }
    return keypoints;
}