//
// Created by Xin Xu on 11/15/17.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "stitcher.h"

using namespace cv;

std::string type2str(int type) {
    std::string r;
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
    r += (chans + '0');
    return r;
}

void findType(Mat& M) {
    std::string ty =  type2str( M.type() );
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

std::vector<Point2d> getWarpCorners(Mat& im, Mat& H) {
    std::vector<Point2d> im_corners, im_corners_warped;
    im_corners.reserve(4);

    // corners before warping
    int h = im.rows;
    int w = im.cols;
    im_corners.push_back(Point2d(0,0));
    im_corners.push_back(Point2d(w,0));
    im_corners.push_back(Point2d(0,h));
    im_corners.push_back(Point2d(w,h));

    perspectiveTransform(im_corners, im_corners_warped, H);

    // 0:(0,0) --- 1:(w,0)
    //    |          |
    // 2:(0,h) --- 3:(w,h)
    return im_corners_warped;
}

Mat createMask(Mat& im) {
    Mat mask = Mat::ones(im.size(), CV_8UC1);
    mask.row(0).setTo(0);
    mask.row(mask.rows-1).setTo(0);
    mask.col(0).setTo(0);
    mask.col(mask.cols-1).setTo(0);
    distanceTransform(mask, mask, CV_DIST_L2, 3);
    double min, max;
    minMaxLoc(im, &min, &max);
    mask /= max;

    std::vector<Mat> singleChannels;
    singleChannels.push_back(mask);
    singleChannels.push_back(mask);
    singleChannels.push_back(mask);
    merge(singleChannels, mask);

    return mask;
}

inline void displayImg(Mat& im) {
    // Create a window for display.
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", im);                // Show our image inside it.
    waitKey(0);
}

// warp image 2 onto image 1 and stitch together
Mat stitchImages(Mat& pano, Mat& image, Mat& H, Mat& pano_mask, Mat& img_mask) {
    int width = pano.cols;
    int height = pano.rows;

    cuda::GpuMat src_gpu, dst_gpu;
    src_gpu.upload(image);

    Mat image_warped;
    image_warped.create(image.size(), image.type());
    dst_gpu.upload(image_warped);

    cuda::warpPerspective(src_gpu, dst_gpu, H, Size(width, height));

    // src_gpu.download(image);
    dst_gpu.download(image_warped);

    Mat bim1, bim2;
    multiply(pano, pano_mask, bim1);
    multiply(image_warped, img_mask, bim2);

    Mat stitch_img;
    divide(bim1 + bim2, pano_mask + img_mask, stitch_img);
    return stitch_img;
}
