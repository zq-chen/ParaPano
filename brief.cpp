//
// Created by Xin Xu on 11/12/17.
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "filter.h"
#include "keyPointDetector.h"
#include "brief.h"

using namespace cv;
using namespace std;

bool hasValidPatch(int h, int w, int row, int col);
float* getPatch(float* im, int begin_row, int begin_col, int w);
Descriptor computeKeypointDescriptor(float* patch, Point* compareA, Point* compareB);

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

void normalize(float* img_ptr, int h, int w) {
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

void outputImageWithKeypoints(Mat& img, vector<Point>& keypoints) {
    vector<Point>::iterator it;
    for (it = keypoints.begin(); it != keypoints.end(); ++it) {
        circle(img, *it, 1, Scalar(0, 0, 255), 1, 8);
    }
    imwrite("../output/keypoints.jpg", img);
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

void cleanPointerArray(float** arr, int num_levels) {
    for (int i = 0; i < num_levels; i++) {
        delete[] arr[i];
    }
    delete[] arr;
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

BriefResult BriefLite(string im_name, Point* compareA, Point* compareB) {

    cout << "Compute BRIEF for image " + im_name << endl;

    Mat im_color = imread(im_name, IMREAD_COLOR);

    // Load as grayscale and convert to float
    Mat im_gray = imread(im_name, IMREAD_GRAYSCALE);
    Mat im;
    im_gray.convertTo(im, CV_32F);
    int h = im.rows;
    int w = im.cols;
    float *im1_ptr = (float*) im.ptr<float>();
    normalize(im1_ptr, h, w);

    // parameters for generating Gaussian Pyramid
    float sigma0 = 1.0;
    float k = sqrt(2);
    int num_levels = 7;
    int levels[7] = {-1, 0, 1, 2, 3, 4, 5};

    float** gaussian_pyramid = createGaussianPyramid(im1_ptr, h, w, sigma0, k, levels, num_levels);
    float** dog_pyramid = createDoGPyramid(gaussian_pyramid, h, w, num_levels);
    cout << "Created DoG Pyramid" << endl;

    // Detect key points
    float th_contrast = 0.03;
    float th_r = 12;
    vector<Point> keypoints = getLocalExtrema(dog_pyramid, num_levels - 1, h, w, th_contrast, th_r);
    printf("Detected %lu key points\n", keypoints.size());

    outputImageWithKeypoints(im_color, keypoints);

    BriefResult brief_result = computeBrief(gaussian_pyramid[0], h, w, keypoints, compareA, compareB);

    // clean up
    cleanPointerArray(gaussian_pyramid, num_levels);
    cleanPointerArray(dog_pyramid, num_levels - 1);
    cout << "Cleaned up Memory" << endl;

    return brief_result;
}