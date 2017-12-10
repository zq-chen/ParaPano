//
// Created by Xin Xu on 11/12/17.
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "brief.h"
#include "cudaMatcher.h"

using namespace cv;
using namespace std;

inline bool isInBound(int r, int c, int h, int w) {
    return r >= 0 && r < h && c >= 0 && c < w;
}

bool hasValidPatch(int h, int w, int row, int col) {
    int r = PATCH_SIZE / 2;
    return isInBound(row - r, col - r, h, w) && isInBound(row + r, col + r,h,w);
}

float* getPatch(float* im, int begin_row, int begin_col, int w) {
    float* patch = new float[PATCH_SIZE * PATCH_SIZE];
    for (int i = begin_row; i < begin_row + PATCH_SIZE; i++) {
        for (int j = begin_col; j < begin_col + PATCH_SIZE; j++) {
            int pi = i - begin_row;
            int pj = j - begin_col;
            patch[pi * PATCH_SIZE + pj] = im[i * w + j];
        }
    }
    return patch;
}

void normalize_img(float* img_ptr, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            img_ptr[i * w + j] /= 255.0;
        }
    }
}

void denormalize(float* img_ptr, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            img_ptr[i * w + j] *= 255.0;
        }
    }
}

void outputImageWithKeypoints(std::string im_path, Mat& img,
                              std::vector<Point>& keypoints) {

    size_t idx = im_path.find_last_of("/\\");
    std::string im_name = im_path.substr(idx+1);
    std::vector<Point>::iterator it;
    for (it = keypoints.begin(); it != keypoints.end(); ++it) {
        circle(img, *it, 1, Scalar(0, 0, 255), 1, 8);
    }
    imwrite("../output/" + im_name + "_keypoints.jpg", img);
    std::cout << "Output image with key points" << std::endl;
}

void outputGaussianImages(float** gaussian_pyramid, int h, int w,
                          int num_levels) {

    for (int i = 0; i < num_levels - 1; i++) {
        denormalize(gaussian_pyramid[i], h, w);
        Mat im(h, w, CV_32F, gaussian_pyramid[i]);
        String imname = "gaussian" + std::to_string(i) + ".jpg";
        imwrite("../output/" + imname, im);
    }
    std::cout << "Output Gaussian Images" << std::endl;
}

void outputDoGImages(float** dog_pyramid, int h, int w, int num_levels) {
    for (int i = 0; i < num_levels - 1; i++) {
        denormalize(dog_pyramid[i], h, w);
        denormalize(dog_pyramid[i], h, w);
        Mat im(h, w, CV_32F, dog_pyramid[i]);
        String imname = "dog" + std::to_string(i) + ".jpg";
        imwrite("../output/" + imname, im);
    }
    std::cout << "Output DoG Images" << std::endl;
}


Descriptor computeKeypointDescriptor(float* patch, Point* compareA,
                                     Point* compareB) {
    Descriptor dscr;
    for (int i = 0; i < NUM_OF_TEST_PAIRS; i++) {
        int x1 = compareA[i].x;
        int y1 = compareA[i].y;
        int x2 = compareB[i].x;
        int y2 = compareB[i].y;
        float a_val = patch[y1 * PATCH_SIZE + x1];
        float b_val = patch[y2 * PATCH_SIZE + x2];
        dscr.set(i, a_val < b_val);
    }
    return dscr;
}

BriefResult computeBrief(float* im, int h, int w, std::vector<Point>& keypoints,
                         Point* compareA, Point* compareB) {

    std::vector<Point> valid_keypoints;
    std::vector<Descriptor> descriptors;
    descriptors.reserve(keypoints.size());
    std::vector<Point>::iterator it;
    for (it = keypoints.begin(); it != keypoints.end(); ++it) {
        int row = it->y;
        int col = it->x;
        if (hasValidPatch(h, w, row, col)) {
            int r = PATCH_SIZE / 2;
            float* patch = getPatch(im, row - r, col - r, w);
            descriptors.push_back(computeKeypointDescriptor(patch, compareA,
                                                            compareB));
            valid_keypoints.push_back(*it);
            delete[] patch;
        }
    }
    return BriefResult(valid_keypoints, descriptors);
}

// int hammingDistance(Descriptor& d1, Descriptor& d2) {
//     return (d1.values ^ d2.values).count();
// }

// float findBestMatch(std::vector<Descriptor>& desc, Descriptor& d,
//                     int& match_idx) {
//     int min = INT_MAX;
//     int second_min = INT_MAX;
//     int min_idx = -1;
//     for (int i = 0; i < desc.size(); i++) {
//         int dist = hammingDistance(desc[i], d);
//         if (dist < min) {
//             second_min = min;
//             min = dist;
//             min_idx = i;
//         } else if (dist < second_min) {
//             second_min = dist;
//         }
//     }
//     match_idx = min_idx;
//     return second_min == 0 ? 1 : float(min) / second_min;
// }

// match desc1 against desc2
// MatchResult briefMatch(std::vector<Descriptor>& desc1,
//                        std::vector<Descriptor>& desc2) {

//     MatchResult match_result;
//     float ratio = 0.8;
//     for (int i = 0; i < desc1.size(); i++) {
//         int match_idx;
//         float r = findBestMatch(desc2, desc1[i], match_idx);
//         if (r < ratio) {
//             match_result.indices1.push_back(i);
//             match_result.indices2.push_back(match_idx);
//         }
//     }
//     return match_result;
// }

// CUDA version of matching key points
MatchResult cudaBriefMatch(std::vector<Descriptor>& desc1, std::vector<Descriptor>& desc2) {
    
    CudaMatcher cudaMatcher;
    cudaMatcher.setup(desc1, desc2);
    return cudaMatcher.findMatch();

    printf("cudaBriefMatch done\n");
}