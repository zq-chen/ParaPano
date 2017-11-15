//
// Created by Xin Xu on 11/15/17.
//
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "stitcher.h"

using namespace cv;
using namespace std;

string type2str(int type) {
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    r += "C";
    r += (chans+'0');
    return r;
}

void findType(Mat& M) {
    string ty =  type2str( M.type() );
    printf("Matrix: %s %dx%d \n", ty.c_str(), M.cols, M.rows );
}

Mat getTranslationMatrix(float x, float y) {
    Mat M = Mat::eye(3, 3, CV_32F);
    M.at<float>(0,2) = x;
    M.at<float>(1,2) = y;
    return M;
}

void convertImg2Float(Mat& im) {
    im.convertTo(im,CV_32FC3);
    im = im / 255;
}

void stitchImages(Mat& im1, Mat& im2, Mat& H) {

    Mat im2_warped;
    vector<Point2d> im2_corners, im2_corners_warped;
    im2_corners.reserve(4);
    int h2 = im2.rows;
    int w2 = im2.cols;
    im2_corners.push_back(Point2d(0,0));
    im2_corners.push_back(Point2d(w2,0));
    im2_corners.push_back(Point2d(0,h2));
    im2_corners.push_back(Point2d(w2,h2));

    perspectiveTransform(im2_corners, im2_corners_warped, H);

    int width = max(im2_corners_warped[1].x, im2_corners_warped[3].x);
    int height = max(abs(im2_corners_warped[0].y - im2_corners_warped[2].y),
                     abs(im2_corners_warped[1].y - im2_corners_warped[3].y));
    int shift_height = abs(im2_corners_warped[1].y);

    Mat M = getTranslationMatrix(0, shift_height);
    Mat mask1 = creatMask(im1);
    Mat mask2 = creatMask(im2);
    warpPerspective(mask1, mask1, M, Size(width, height));
    warpPerspective(mask2, mask2, M*H, Size(width, height));
    warpPerspective(im1, im1, M, Size(width, height));
    warpPerspective(im2, im2_warped, M*H, Size(width, height));

    convertImg2Float(im1);
    convertImg2Float(im2_warped);

    Mat bim1, bim2;
    multiply(im1, mask1, bim1);
    multiply(im2_warped, mask2, bim2);
    Mat panoImg;
    divide(bim1 + bim2, mask1 + mask2, panoImg);

    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", panoImg);              // Show our image inside it.
    waitKey(0);
    imwrite("../output/panorama.jpg", panoImg);
}

Mat creatMask(Mat& im) {
    Mat mask = Mat::ones(im.size(), CV_8UC1);
    mask.row(0).setTo(0);
    mask.row(mask.rows-1).setTo(0);
    mask.col(0).setTo(0);
    mask.col(mask.cols-1).setTo(0);
    distanceTransform(mask, mask, CV_DIST_L2, 3);
    double min, max;
    minMaxLoc(im, &min, &max);
    mask /= max;

    vector<Mat> singleChannels;
    singleChannels.push_back(mask);
    singleChannels.push_back(mask);
    singleChannels.push_back(mask);
    merge(singleChannels, mask);

    return mask;
}
