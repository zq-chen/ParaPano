//
// Created by Xin Xu on 11/10/17.
//

#ifndef PARAPANO_KEYPOINTDETECTOR_H
#define PARAPANO_KEYPOINTDETECTOR_H

#include <opencv2/core/core.hpp>
#include <vector>

float** createDoGPyramid(float** gaussian_pyramid, int h, int w, int num_levels);
std::vector<cv::Point> getLocalExtrema(float** dog_pyramid, int num_levels, int h, int w, float th_contrast, float th_r);

#endif //PARAPANO_KEYPOINTDETECTOR_H