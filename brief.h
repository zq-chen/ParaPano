//
// Created by Xin Xu on 11/12/17.
//

#pragma once

#include <bitset>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "bitarray.h"

#define PATCH_SIZE 9
#define NUM_OF_TEST_PAIRS 256

struct Descriptor {
  uint64_t num0;
  uint64_t num1;
  uint64_t num2;
  uint64_t num3;

  Descriptor() {
    num0 = 0;
    num1 = 0;
    num2 = 0;
    num3 = 0;
  }

  /* valid pos: 0 <= pos <= 255 */
  void set(int pos, bool cond) {
    if (pos < 0 || pos > 255) {
      printf("Invalid pos\n");
      return;
    }
    if (cond) {
      int id = pos / 64;
      int real_pos = pos % 64;

      /* Bug fixed: must use 1L here! 
       * 1 is considered int
       * so 1 << 31 is undefined
       */
      uint64_t tmp = 1L << real_pos;
      switch (id) {
        case 0:
          num0 |= tmp;
          break;
        case 1:
          num1 |= tmp;
          break;
        case 2:
          num2 |= tmp;
          break;
        case 3:
          num3 |= tmp;
          break;
      }
    }
  }

  void print() {
    std::cout << std::bitset<64>(num3) << std::bitset<64>(num2);
    std::cout << std::bitset<64>(num1) << std::bitset<64>(num0) << std::endl;
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

MatchResult cudaBriefMatch(std::vector<Descriptor>& desc1, std::vector<Descriptor>& desc2);

BriefResult computeBrief(float* im, int h, int w,
                         std::vector<cv::Point>& keypoints, cv::Point* compareA,
                         cv::Point* compareB);

void outputImageWithKeypoints(std::string im_path, cv::Mat& img,
                              std::vector<cv::Point>& keypoints);

void outputDoGImages(float** dog_pyramid, int h, int w, int num_levels);
void normalize_img(float* img_ptr, int h, int w);
