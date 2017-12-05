//
// Created by Xin Xu on 11/15/17.
//

#pragma once

#include "opencv2/opencv.hpp"

cv::Mat createMask(cv::Mat& im);
cv::Mat stitchImages(cv::Mat& pano, cv::Mat& image, cv::Mat& H, 
	cv::Mat& pano_mask, cv::Mat& img_mask);
std::vector<cv::Point2d> getWarpCorners(cv::Mat& im, cv::Mat& H);
cv::Mat getTranslationMatrix(float x, float y);

void convertImg2Float(cv::Mat& im);