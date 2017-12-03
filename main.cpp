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

extern const int num_images = 2;

int main(int argc, char** argv) {

    Util util;

    std::string im_names[2] = {"../data/incline_L.png","../data/incline_R.png"};

//    string im_names[4] = {"../data/mountain1.jpg", "../data/mountain2.jpg", 
//    "../data/mountain3.jpg", "../data/mountain4.jpg"};
//    int num_images = 4;

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

    // compute BRIEF for keypoints
    std::vector<BriefResult> brief_results;
    brief_results.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        BriefResult brief_result = util.BriefLite(im_names[i],
                                                  compareA, compareB);
        brief_results.push_back(brief_result);
    }

    std::vector<Mat> homographies;
    homographies.reserve(num_images-1);
    for (int i = 0; i < num_images-1; i++) {
        Mat H = util.computeHomography(im_names[i], im_names[i+1],
                                       brief_results[i], brief_results[i+1]);
        homographies.push_back(H);
    }

    // Perform image stitching
    util.stitch(images, homographies);

    util.printTiming();

    return 0;
}
