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

#include "Util.h"

using namespace cv;

extern const int num_images = 6;

int main(int argc, char** argv) {

    double total_time_elapsed = 0.0;
    clock_t total_start = clock();

    double IO_elapsed = 0.0;
    clock_t IO_start = clock();

    Util util;

   //std::string im_names[1] = {"../data/mountainR.png"};

   std::string im_names[num_images];
   for (int i = 0; i < num_images; i++) {
       im_names[i] = "../data/campus/fence" + std::to_string(i+1) + ".jpeg";
   }

    std::vector<Mat> images;
    images.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        Mat im;
        if (!util.readImage(im_names[i], im)) {
            return -1;
        }
        convertImg2Float(im);
        images.push_back(im);
    }

    // read in test pattern points to compute BRIEF
    Point* compareA = NULL;
    Point* compareB = NULL;
    std::string test_pattern_filename = "../data/testPattern.txt";
    util.readTestPattern(compareA, compareB, test_pattern_filename);

    IO_elapsed = util.get_time_elapsed(IO_start);    

    // compute BRIEF for keypoints
    std::vector<BriefResult> brief_results;
    brief_results.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        BriefResult brief_result = util.BriefLite(im_names[i],
                                                  compareA, compareB);
        brief_results.push_back(brief_result);
    }

    // compute homographies relative to the first image
    double compute_homography_elapsed = 0.0;
    clock_t homography_start = clock();

    std::vector<Mat> homographies;
    homographies.reserve(num_images);
    Mat identity = Mat::eye(3, 3, CV_32F);
    homographies.push_back(identity);
    for (int i = 1; i < num_images; i++) {

        // compute transformation between image(i) and image(i-1)
        Mat H = util.computeHomography(im_names[i-1], im_names[i],
                                       brief_results[i-1], brief_results[i]);

        // Compute H(i) * H(i-1) * ... * H(1)
        H = homographies[i-1] * H;
        homographies.push_back(H);
    }
    std::cout << "Computed homographies" << std::endl;

    // find inverse of center image
    int center_image_idx = (num_images - 1) / 2;
    Mat center_inverse = homographies[center_image_idx].inv();

    // apply center homo inverse and compute panorama size
    double xMin, yMin = INT_MAX;
    double xMax, yMax = 0;
    for (int i = 0; i < homographies.size(); i++) {
        homographies[i] = center_inverse * homographies[i];
        std::vector<Point2d> corners = getWarpCorners(images[i], homographies[i]);
        for (int j = 0; j < corners.size(); j++) {
            xMin = std::min(xMin, corners[j].x);
            xMax = std::max(xMax, corners[j].x);
            yMin = std::min(yMin, corners[j].y);
            yMax = std::max(yMax, corners[j].y);
        }
    }
    

    // shift the panorama if warped images are out of boundaries
    double shiftX = -xMin;
    double shiftY = -yMin;
    Mat transM = getTranslationMatrix(shiftX, shiftY);

    // initialize empty panorama
    int width = std::round(xMax - xMin);
    int height = std::round(yMax - yMin);
    Mat panorama = Mat::zeros(height, width, CV_32F);

    // apply translation to homographies
    for (int i = 0; i < homographies.size(); i++) {
        homographies[i] = transM * homographies[i];
        // normalze homography matrix
        homographies[i] = homographies[i]/homographies[i].at<float>(2,2);
    }
    std::cout << "Adjusted Panorama to Center Image" << std::endl;

    compute_homography_elapsed = util.get_time_elapsed(homography_start);

    // Perform image stitching
    util.stitch(images, homographies, width, height);

    total_time_elapsed = util.get_time_elapsed(total_start);
    printf("Total Time is %.2f\n", total_time_elapsed);
    printf("IO Time is %.2f\n", IO_elapsed);
    printf("Compute Homography Time is %.2f\n", compute_homography_elapsed);

    util.printTiming();

    return 0;
}
