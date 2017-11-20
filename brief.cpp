//
// Created by Xin Xu on 11/12/17.
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <limits>
//#include "filter.h"
//#include "keyPointDetector.h"
#include "brief.h"

using namespace cv;
using namespace std;

inline bool isInBound(int r, int c, int h, int w) {
    return r >= 0 && r < h && c >= 0 && c < w;
}

bool hasValidPatch(int h, int w, int row, int col) {
    int r = PATCH_SIZE/2;
    return isInBound(row - r, col - r, h, w) && isInBound(row + r, col + r, h, w);
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

void outputImageWithKeypoints(string im_path, Mat& img, vector<Point>& keypoints) {
    size_t idx = im_path.find_last_of("/\\");
    string im_name = im_path.substr(idx+1);
    vector<Point>::iterator it;
    for (it = keypoints.begin(); it != keypoints.end(); ++it) {
        circle(img, *it, 1, Scalar(0, 0, 255), 1, 8);
    }
    imwrite("../output/" + im_name + "_keypoints.jpg", img);
    cout << "Output image with key points" << endl;
}

void outputGaussianImages(float** gaussian_pyramid, int h, int w, int num_levels) {
    for (int i = 0; i < num_levels - 1; i++) {
        denormalize(gaussian_pyramid[i], h, w);
        Mat im(h, w, CV_32F, gaussian_pyramid[i]);
        String imname = "gaussian" + to_string(i) + ".jpg";
        imwrite("../output/" + imname, im);
    }
    cout << "Output Gaussian Images" << endl;
}

void outputDoGImages(float** dog_pyramid, int h, int w, int num_levels) {
    for (int i = 0; i < num_levels - 1; i++) {
        denormalize(dog_pyramid[i], h, w);
        denormalize(dog_pyramid[i], h, w);
        Mat im(h, w, CV_32F, dog_pyramid[i]);
        String imname = "dog" + to_string(i) + ".jpg";
        imwrite("../output/" + imname, im);
    }
    cout << "Output DoG Images" << endl;
}


Descriptor computeKeypointDescriptor(float* patch, Point* compareA, Point* compareB) {
    Descriptor dscr;
    for (int i = 0; i < NUM_OF_TEST_PAIRS; i++) {
        int x1 = compareA[i].x;
        int y1 = compareA[i].y;
        int x2 = compareB[i].x;
        int y2 = compareB[i].y;
        float a_val = patch[y1 * PATCH_SIZE + x1];
        float b_val = patch[y2 * PATCH_SIZE + x2];
        dscr.values[i] = a_val < b_val? 1 : 0;
    }
    return dscr;
}

BriefResult computeBrief(float* im, int h, int w, vector<Point>& keypoints, Point* compareA, Point* compareB) {
    vector<Point> valid_keypoints;
    vector<Descriptor> descriptors;
    descriptors.reserve(keypoints.size());
    vector<Point>::iterator it;
    for (it = keypoints.begin(); it != keypoints.end(); ++it) {
        int row = it->y;
        int col = it->x;
        if (hasValidPatch(h, w, row, col)) {
            int r = PATCH_SIZE/2;
            float* patch = getPatch(im, row-r, col-r, w);
            descriptors.push_back(computeKeypointDescriptor(patch, compareA, compareB));
            valid_keypoints.push_back(*it);
            delete[] patch;
        }
    }
    return BriefResult(valid_keypoints, descriptors);
}


int hammingDistance(Descriptor& d1, Descriptor& d2) {
    int dist = 0;
    for (int i = 0; i < NUM_OF_TEST_PAIRS; i++) {
        if (d1.values[i] != d2.values[i]) {
            dist += 1;
        }
    }
    return dist;
}

float findBestMatch(vector<Descriptor>& desc, Descriptor& d, int& match_idx) {
    int min = INT_MAX;
    int second_min = INT_MAX;
    int min_idx = -1;
    for (int i = 0; i < desc.size(); i++) {
        int dist = hammingDistance(desc[i], d);
        if (dist < min) {
            second_min = min;
            min = dist;
            min_idx = i;
        } else if (dist < second_min) {
            second_min = dist;
        }
    }
    match_idx = min_idx;
    return second_min == 0? 1:float(min)/second_min;
}

// match desc1 against desc2
MatchResult briefMatch(vector<Descriptor>& desc1, vector<Descriptor>& desc2) {
    MatchResult match_result;
    float ratio = 0.8;
    for (int i = 0; i < desc1.size(); i++) {
        int match_idx;
        float r = findBestMatch(desc2, desc1[i], match_idx);
        if (r < ratio) {
            match_result.indices1.push_back(i);
            match_result.indices2.push_back(match_idx);
        }
    }
    return match_result;
}