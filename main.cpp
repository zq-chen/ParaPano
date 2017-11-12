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

int main(int argc, char** argv) {

    string im1_name = "../data/model_chickenbroth.jpg";
    if( argc > 1) {
        im1_name = argv[1];
    }

    // read in test pattern points to compute BRIEF
    Point* compareA = NULL;
    Point* compareB = NULL;
    string test_pattern_filename = "../data/testPattern.txt";
    readTestPattern(compareA, compareB, test_pattern_filename);

    // compute BRIEF for keypoints
    BriefResult brief_result = BriefLite(im1_name, compareA, compareB);



//    for (int i = 0; i < brief_result.keypoints.size(); i++) {
//        printf("Point %d: ", i);
//        for (int j = 0; j < NUM_OF_TEST_PAIRS; j++) {
//            printf("%d ", brief_result.descriptors[i].values[j]);
//        }
//        cout << endl;
//    }
    return 0;
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