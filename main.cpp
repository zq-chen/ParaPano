//
// Created by Xin Xu on 11/9/17.
//
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include "filter.h"
#include "keyPointDetector.h"
#include "brief.h"
#include "stitcher.h"


using namespace cv;
using namespace std;

int readTestPattern(Point*& compareA, Point*& compareB, string test_pattern_filename);
void plotMatches(string im1_name, string im2_name, vector<Point>& pts1, vector<Point>& pts2, MatchResult& match);
Mat computeHomography(string im1_name, string im2_name, BriefResult brief_result1, BriefResult brief_result2);
BriefResult BriefLite(string im_name, Point* compareA, Point* compareB);
void displayImg(Mat& im);

clock_t gaussian_pyramid_start;
double gaussian_pyramid_elapsed = 0.0;

clock_t dog_pyramid_start;
double dog_pyramid_elapsed = 0.0;

clock_t keypoint_detection_start;
double keypoint_detection_elapsed = 0.0;

clock_t compute_brief_start;
double compute_brief_elapsed = 0.0;

clock_t find_match_start;
double find_match_elapsed = 0.0;

clock_t compute_homography_start;
double compute_homography_elapsed = 0.0;

clock_t stitching_start;
double stitching_elapsed;

static inline double get_time_elapsed(clock_t& start)
{
    return (clock() - start)/(double)CLOCKS_PER_SEC;
}

bool ReadImage(string im_name, Mat& im) {
    im = imread(im_name, IMREAD_COLOR);
    if(!im.data ) {
        cout <<  "Could not open or find the image " + im_name << endl ;
        return false;
    }
    return true;
}

void cleanPointerArray(float** arr, int num_levels) {
    for (int i = 0; i < num_levels; i++) {
        delete[] arr[i];
    }
    delete[] arr;
}

int main(int argc, char** argv) {

    int num_images = 2;
    string im_names[2] = {"../data/incline_L.png", "../data/incline_R.png"};

//    string im_names[4] = {"../data/mountain1.jpg", "../data/mountain2.jpg", "../data/mountain3.jpg", "../data/mountain4.jpg"};
//    int num_images = 4;

    vector<Mat> images;
    images.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        Mat im;
        if (!ReadImage(im_names[i], im)) {
            return -1;
        }
        convertImg2Float(im);
        images.push_back(im);
    }

    // read in test pattern points to compute BRIEF
    Point* compareA = NULL;
    Point* compareB = NULL;
    string test_pattern_filename = "../data/testPattern.txt";
    readTestPattern(compareA, compareB, test_pattern_filename);

    // compute BRIEF for keypoints
    vector<BriefResult> brief_results;
    brief_results.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        BriefResult brief_result = BriefLite(im_names[i], compareA, compareB);
        brief_results.push_back(brief_result);
    }

    vector<Mat> homographies;
    homographies.reserve(num_images-1);
    for (int i = 0; i < num_images-1; i++) {
        Mat H = computeHomography(im_names[i], im_names[i+1], brief_results[i], brief_results[i+1]);
        homographies.push_back(H);
    }

    stitching_start = clock();
    Mat prev_img = images[0];
    Mat prev_H = Mat::eye(3, 3, CV_32F);
    Mat mask1 = creatMask(prev_img);
    for (int i = 1; i < num_images; i++) {

        Mat H = prev_H * homographies[i-1];
        H = H/H.at<float>(2,2); // normalze
        Mat stitch_img = stitchImages(prev_img, images[i], mask1, H, prev_H);
        prev_img = stitch_img;

        string output_name = "../output/stitch_" + to_string(i) + ".jpg";
        imwrite(output_name, stitch_img*255);

    }
    stitching_elapsed = get_time_elapsed(stitching_start);

    // displayImg(prev_img);
    imwrite("../output/panorama.jpg", prev_img*255);

    printf("Compute Gaussian Pyramid: %.2f\n", gaussian_pyramid_elapsed);
    printf("Compute DoG Pyramid: %.2f\n", dog_pyramid_elapsed);
    printf("Detect Keypoints: %.2f\n", keypoint_detection_elapsed);
    printf("Compute BRIEF Descriptor: %.2f\n", compute_brief_elapsed);
    printf("Match keypoint descriptors: %.2f\n", find_match_elapsed);
    printf("Compute Homography: %.2f\n", compute_homography_elapsed);
    printf("Stitch Images: %.2f\n", stitching_elapsed);

    return 0;
}


Mat computeHomography(string im1_name, string im2_name, BriefResult brief_result1, BriefResult brief_result2) {

    find_match_start = clock();
    MatchResult match = briefMatch(brief_result1.descriptors, brief_result2.descriptors);
    // plotMatches(im1_name, im2_name, brief_result1.keypoints, brief_result2.keypoints, match);
    vector<Point> pts1;
    pts1.reserve(match.indices1.size());
    vector<Point> pts2;
    pts2.reserve(match.indices2.size());
    for (int i = 0; i < match.indices1.size(); i++) {
        int idx1 = match.indices1[i];
        int idx2 = match.indices2[i];
        pts1.push_back(brief_result1.keypoints[idx1]);
        pts2.push_back(brief_result2.keypoints[idx2]);
    }
    find_match_elapsed += get_time_elapsed(find_match_start);


    compute_homography_start = clock();
    Mat H = findHomography(pts2, pts1, RANSAC, 4.0);
    compute_homography_elapsed += get_time_elapsed(compute_homography_start);

    H.convertTo(H,CV_32F);
    return H;
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
    normalize_img(im1_ptr, h, w);

    // parameters for generating Gaussian Pyramid
    float sigma0 = 1.0;
    float k = sqrt(2);
    int num_levels = 7;
    int levels[7] = {-1, 0, 1, 2, 3, 4, 5};

    gaussian_pyramid_start = clock();
    float** gaussian_pyramid = createGaussianPyramid(im1_ptr, h, w, sigma0, k, levels, num_levels);
    gaussian_pyramid_elapsed += get_time_elapsed(gaussian_pyramid_start);


    dog_pyramid_start = clock();
    float** dog_pyramid = createDoGPyramid(gaussian_pyramid, h, w, num_levels);
    // outputGaussianImages(gaussian_pyramid, h, w, num_levels);
    // outputDoGImages(dog_pyramid, h, w, num_levels);
    cout << "Created DoG Pyramid" << endl;
    dog_pyramid_elapsed += get_time_elapsed(dog_pyramid_start);

    keypoint_detection_start = clock();
    // Detect key points
    float th_contrast = 0.03;
    float th_r = 12;
    vector<Point> keypoints = getLocalExtrema(dog_pyramid, num_levels - 1, h, w, th_contrast, th_r);
    printf("Detected %lu key points\n", keypoints.size());
    keypoint_detection_elapsed += get_time_elapsed(keypoint_detection_start);

    outputImageWithKeypoints(im_name, im_color, keypoints);

    compute_brief_start = clock();
    BriefResult brief_result = computeBrief(gaussian_pyramid[0], h, w, keypoints, compareA, compareB);
    compute_brief_elapsed += get_time_elapsed(compute_brief_start);

    // clean up
    cleanPointerArray(gaussian_pyramid, num_levels);
    cleanPointerArray(dog_pyramid, num_levels - 1);
    cout << "Cleaned up Memory" << endl;

    return brief_result;
}

void plotMatches(string im1_name, string im2_name, vector<Point>& pts1, vector<Point>& pts2, MatchResult& match) {
    Mat im1 = imread(im1_name, IMREAD_COLOR);
    Mat im2 = imread(im2_name, IMREAD_COLOR);
    int h1 = im1.rows;
    int w1 = im1.cols;
    int h2 = im2.rows;
    int w2 = im2.cols;
    int width = w1 + w2;
    int height = max(h1, h2);
    Mat grid(height, width, CV_8UC3, Scalar(0,0,0));
    im1.copyTo(grid(Rect(0,0,w1,h1)));
    im2.copyTo(grid(Rect(w1,0,w2,h2)));

    for (int i = 0; i < match.indices1.size(); i++) {
        Point p1 = pts1[match.indices1[i]];
        Point p2 = pts2[match.indices2[i]];
        line(grid, p1, Point(p2.x+w1, p2.y), Scalar(0, 0, 255));
    }

    cout << "Output Match Image" << endl;
    imwrite("../output/match.jpg", grid);
}

int readTestPattern(Point*& compareA, Point*& compareB, string test_pattern_filename) {
    ifstream infile(test_pattern_filename);
    int num_test_pairs;
    infile >> num_test_pairs;
    compareA = new Point[num_test_pairs];
    compareB = new Point[num_test_pairs];
    int x1, y1, x2, y2;
    for (int i = 0; i < num_test_pairs; i++) {
        infile >> x1 >> y1 >> x2 >> y2;
        compareA[i] = Point(x1, y1);
        compareB[i] = Point(x2, y2);
    }
    return num_test_pairs;
}

void printImage(float* img, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            cout<<img[i * w + j]<<" ";
        }
        cout<<endl;
    }
}

void displayImg(Mat& im) {
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", im);                // Show our image inside it.
    waitKey(0);
}