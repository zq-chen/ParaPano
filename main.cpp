//
// Created by Xin Xu on 11/9/17.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "brief.h"

using namespace cv;
using namespace std;

int readTestPattern(Point*& compareA, Point*& compareB, string test_pattern_filename);
void plotMatches(string im1_name, string im2_name, vector<Point>& pts1, vector<Point>& pts2, MatchResult& match);

int main(int argc, char** argv) {

    string im1_name = "../data/incline_L.png";
    string im2_name = "../data/incline_R.png";
    if( argc > 2) {
        im1_name = argv[1];
        im2_name = argv[2];
    }

    Mat im1 = imread(im1_name, IMREAD_COLOR);
    Mat im2 = imread(im2_name, IMREAD_COLOR);
    if(!im1.data ) {
        cout <<  "Could not open or find the image " + im1_name << endl ;
        return -1;
    }
    if(!im2.data ) {
        cout <<  "Could not open or find the image " + im2_name << endl ;
        return -1;
    }

    // read in test pattern points to compute BRIEF
    Point* compareA = NULL;
    Point* compareB = NULL;
    string test_pattern_filename = "../data/testPattern.txt";
    readTestPattern(compareA, compareB, test_pattern_filename);

    // compute BRIEF for keypoints
    BriefResult brief_result1 = BriefLite(im1_name, compareA, compareB);
    BriefResult brief_result2 = BriefLite(im2_name, compareA, compareB);

    MatchResult match = briefMatch(brief_result1.descriptors, brief_result2.descriptors);
    plotMatches(im1_name, im2_name, brief_result1.keypoints, brief_result2.keypoints, match);

    

    return 0;
}

void plotMatches(string im1_name, string im2_name, vector<Point>& pts1, vector<Point>& pts2, MatchResult& match) {
    Mat im1 = imread(im1_name, IMREAD_COLOR);
    Mat im2 = imread(im2_name, IMREAD_COLOR);
    int h1 = im1.rows;
    int w1 = im1.cols;
    int h2 = im2.rows;
    int w2 = im2.cols;
    int width = w1 + w2;
    int height = max(h1, h2);
    Mat grid(height, width, CV_8UC3, Scalar(0,0,0));
    im1.copyTo(grid(Rect(0,0,w1,h1)));
    im2.copyTo(grid(Rect(w1,0,w2,h2)));

    for (int i = 0; i < match.indices1.size(); i++) {
        Point p1 = pts1[match.indices1[i]];
        Point p2 = pts2[match.indices2[i]];
        line(grid, p1, Point(p2.x+w1, p2.y), Scalar(0, 0, 255));
    }

    cout << "Output Match Image" << endl;
    imwrite("../output/match.jpg", grid);

//    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
//    imshow( "Display window", grid);                // Show our image inside it.
//    waitKey(0);
}

int readTestPattern(Point*& compareA, Point*& compareB, string test_pattern_filename) {
    ifstream infile(test_pattern_filename);
    int num_test_pairs;
    infile >> num_test_pairs;
    compareA = new Point[num_test_pairs];
    compareB = new Point[num_test_pairs];
    int x1, y1, x2, y2;
    for (int i = 0; i < num_test_pairs; i++) {
        infile >> x1 >> y1 >> x2 >> y2;
        compareA[i] = Point(x1, y1);
        compareB[i] = Point(x2, y2);
    }
    return num_test_pairs;
}

void printImage(float* img, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            cout<<img[i * w + j]<<" ";
        }
        cout<<endl;
    }
}

void displayImg(float* img_ptr, int h, int w) {
//  normalize(img_ptr, h, w);
    Mat im (h, w, CV_32F, img_ptr);
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", im);                // Show our image inside it.
    waitKey(0);
}