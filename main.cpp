//
// Created by Xin Xu on 11/9/17.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gaussian.h"

using namespace cv;
using namespace std;

void display(Mat& im);

int main(int argc, char** argv) {

    cout << "Create gaussian filter" << endl;
    float* gaussian = createGaussianFilter(3, 3, 0.5);
    printGaussianFilter(gaussian, 3, 3);

    String im1_name( "../data/incline_L.png");
    if( argc > 1) {
        im1_name = argv[1];
    }

    Mat im1_color = imread(im1_name, IMREAD_COLOR);
    Mat im1_gray;
    cvtColor(im1_color,im1_gray,CV_RGB2GRAY);

    display(im1_gray);
    return 0;
}

void display(Mat& im) {
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", im);                // Show our image inside it.
    waitKey(0);
}
