//
// Created by zhuoqunc on 12/02/17.
//

#pragma once

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "brief.h"
#include "filter.h"
#include "keyPointDetector.h"
#include "stitcher.h"

class Util
{
public:
    Util();

    virtual ~Util();

    double get_time_elapsed(clock_t& start);

    bool readImage(std::string im_name, cv::Mat& im);

    void cleanPointerArray(float** arr, int num_levels);

    int readTestPattern(cv::Point*& compareA, cv::Point*& compareB,
                        std::string test_pattern_filename);

    void plotMatches(std::string im1_name, std::string im2_name, 
                     std::vector<cv::Point>& pts1, std::vector<cv::Point>& pts2,
                     MatchResult& match);

    cv::Mat computeHomography(std::string im1_name, std::string im2_name, 
                          BriefResult brief_result1, BriefResult brief_result2);

    BriefResult BriefLite(std::string im_name, cv::Point* compareA,
                          cv::Point* compareB);

    void stitch(std::vector<cv::Mat> images, std::vector<cv::Mat> homographies);

    void printImage(float* img, int h, int w) const;

    void displayImg(cv::Mat& im) const;

    void printTiming() const;

private:
    clock_t gaussian_pyramid_start;
    double gaussian_pyramid_elapsed;

    clock_t dog_pyramid_start;
    double dog_pyramid_elapsed;

    clock_t keypoint_detection_start;
    double keypoint_detection_elapsed;

    clock_t compute_brief_start;
    double compute_brief_elapsed;

    clock_t find_match_start;
    double find_match_elapsed;

    clock_t compute_homography_start;
    double compute_homography_elapsed;

    clock_t stitching_start;
    double stitching_elapsed;
};


