//
// Created by Xin Xu on 11/15/17.
//

#ifndef PARAPANO_STITCHER_H
#define PARAPANO_STITCHER_H

#include "opencv2/opencv.hpp"

void stitchImages(cv::Mat& im1, cv::Mat& im2, cv::Mat& H);
cv::Mat creatMask(cv::Mat& im);

#endif //PARAPANO_STITCHER_H
