//
// Created by Xin Xu on 11/9/17.
//

#ifndef PARAPANO_GAUSSIAN_H
#define PARAPANO_GAUSSIAN_H

float** createGaussianPyramid(float* img, int h, int w, float sigma0, float k, int* levels, int num_levels);

#endif //PARAPANO_GAUSSIAN_H
