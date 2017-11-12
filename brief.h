//
// Created by Xin Xu on 11/12/17.
//

#ifndef PARAPANO_BRIEF_H
#define PARAPANO_BRIEF_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#define PATCH_SIZE 9
#define NUM_OF_TEST_PAIRS 256

struct Descriptor {
    int values[NUM_OF_TEST_PAIRS];
};

struct BriefResult {
    std::vector<cv::Point> keypoints;
    std::vector<Descriptor> descriptors;
    BriefResult(std::vector<cv::Point> k, std::vector<Descriptor> d): keypoints(k), descriptors(d) {}
};


BriefResult BriefLite(std::string im_name, cv::Point* compareA, cv::Point* compareB);

#endif //PARAPANO_BRIEF_H