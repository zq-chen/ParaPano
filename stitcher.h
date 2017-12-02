//
// Created by Xin Xu on 11/15/17.
//

#pragma once

#include "opencv2/opencv.hpp"

cv::Mat stitchImages(cv::Mat& im1, cv::Mat& im2, cv::Mat& mask1, cv::Mat& H,
	                 cv::Mat& prev_H);

cv::Mat creatMask(cv::Mat& im);

void convertImg2Float(cv::Mat& im);