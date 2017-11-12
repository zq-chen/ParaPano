//
// Created by Xin Xu on 11/9/17.
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


using namespace cv;
using namespace std;

void normalize(float* img_ptr, int h, int w);
void denormalize(float* img_ptr, int h, int w);
void displayImg(float* img_ptr, int h, int w);
void cleanPointerArray(float** arr, int num_levels);
void outputGaussianImages(float** gaussian_pyramid, int h, int w, int num_levels);
void outputDoGImages(float** dog_pyramid, int h, int w, int num_levels);
void outputImageWithKeypoints(Mat& img, vector<Point>& keypoints);
void printImage(float* img, int h, int w);

int main(int argc, char** argv) {

    String im1_name( "../data/model_chickenbroth.jpg");
    if( argc > 1) {
        im1_name = argv[1];
    }

    // Load as grayscale and convert to float
    Mat im_color = imread(im1_name, IMREAD_COLOR);
    Mat im_gray = imread(im1_name, IMREAD_GRAYSCALE);
    Mat im;
    im_gray.convertTo(im, CV_32F);

    // parameters for generating Gaussian Pyramid
    float sigma0 = 1.0;
    float k = sqrt(2);
    int num_levels = 7;
    int levels[7] = {-1, 0, 1, 2, 3, 4, 5};
    int h = im.rows;
    int w = im.cols;
    float *im1_ptr = (float*) im.ptr<float>();
    normalize(im1_ptr, h, w);

    // int fsize = 3;
    float** gaussian_pyramid = createGaussianPyramid(im1_ptr, h, w, sigma0, k, levels, num_levels);
    float** dog_pyramid = createDoGPyramid(gaussian_pyramid, h, w, num_levels);
    cout << "Created DoG Pyramid" << endl;

    // Detect key points
    float th_contrast = 0.03;
    float th_r = 12;
    vector<Point> keypoints = getLocalExtrema(dog_pyramid, num_levels - 1, h, w, th_contrast, th_r);
    printf("Detected %lu key points\n", keypoints.size());

    outputImageWithKeypoints(im_color, keypoints);

    // clean up
    cleanPointerArray(gaussian_pyramid, num_levels);
    cleanPointerArray(dog_pyramid, num_levels - 1);
    cout << "Cleaned up Memory" << endl;

    return 0;
}

void printImage(float* img, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            cout<<img[i * w + j]<<" ";
        }
        cout<<endl;
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

void displayImg(float* img_ptr, int h, int w) {
//  normalize(img_ptr, h, w);
    Mat im (h, w, CV_32F, img_ptr);
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", im);                // Show our image inside it.
    waitKey(0);
}