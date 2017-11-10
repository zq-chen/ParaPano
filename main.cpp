//
// Created by Xin Xu on 11/9/17.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gaussian.h"

using namespace std;

int main() {

    cout << "Create gaussian filter" << endl;
    float* gaussian = createGaussianFilter(3, 3, 0.5);
    printGaussianFilter(gaussian, 3, 3);

    return 0;
}
