//
// Created by Xin Xu on 11/12/17.
//

#pragma once

// #include <bitset>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "bitarray.h"

#define PATCH_SIZE 9
#define NUM_OF_TEST_PAIRS 256

struct Descriptor {
    // std::bitset<NUM_OF_TEST_PAIRS> values;
  //BitArray values;
  int num_cells;
  int64_t num0;
  int64_t num1;
  int64_t num2;
  int64_t num3;

  Descriptor() {
    num_cells = 4;
    num0 = 0x0LL;
    num1 = 0x0LL;
    num2 = 0x0LL;
    num3 = 0x0LL;
  }

  /* valid pos: 0 <= pos <= 255 */
  void set(int pos, bool cond) {
    if (pos < 0 || pos > 255) {
      printf("Invalid pos\n");
      return;
    }
    if (cond) {
      int id = pos / 4;
      int real_pos = pos % 64;
      int64_t tmp = 1 << real_pos;
      switch (id) {
        case 0:
          num0 |= tmp;
        case 1:
          num1 |= tmp;
        case 2:
          num2 |= tmp;
        case 3:
          num3 |= tmp;
      }
    }
  }
};

struct BriefResult {
    std::vector<cv::Point> keypoints;
    std::vector<Descriptor> descriptors;
    BriefResult(std::vector<cv::Point> k, std::vector<Descriptor> d):
                                          keypoints(k), descriptors(d) {}
};

struct MatchResult {
    std::vector<int> indices1;
    std::vector<int> indices2;
};

//BriefResult BriefLite(std::string im_name, cv::Point* compareA,
// cv::Point* compareB);
// MatchResult briefMatch(std::vector<Descriptor>& desc1,
//                        std::vector<Descriptor>& desc2);

MatchResult cudaBriefMatch(std::vector<Descriptor>& desc1,
                       std::vector<Descriptor>& desc2);

BriefResult computeBrief(float* im, int h, int w,
                         std::vector<cv::Point>& keypoints, cv::Point* compareA,
                         cv::Point* compareB);

void outputImageWithKeypoints(std::string im_path, cv::Mat& img,
                              std::vector<cv::Point>& keypoints);

void normalize_img(float* img_ptr, int h, int w);
