//
// Created by Xin Xu on 11/9/17.
//

#pragma once

float** createGaussianPyramid(float* img, int h, int w, float sigma0, float k,
	                          int* levels, int num_levels);
