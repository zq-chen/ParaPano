# ParaPano

Parallel image stitching using CUDA.

## Summary

We are going to parallelize an image stitching program that aligns a set of images and stitch them together to
produce a panorama. We will compare the speedup and quality of our parallel algorithm with the sequential version.

## Progress

We have finished the sequential implementation of the image stitching program and obtained satisfactory results with
a few test images. Some output examples of our program are presented here:

<img src="https://user-images.githubusercontent.com/16803685/33004052-83c71a2a-cd8c-11e7-8a25-9c805f4a500f.png" alt="img0" width="800" align="middle" />

<img src="https://user-images.githubusercontent.com/16803685/33004054-8af269f8-cd8c-11e7-9255-c1e12e2e1d84.png" alt="img0" width="800" align="middle" />

<img src="https://user-images.githubusercontent.com/16803685/33004058-93a359b8-cd8c-11e7-9d2e-b4e1fd1f35a9.png" alt="img0" width="800" align="middle" />


The program consists of several stages:

1) Convolve a set of Gaussian filters on the image.

2) Build a Difference of Gaussian (GoD) pyramid.

3) Detect key points.

4) Compute Brief descriptors for key point.

5) Match key points in two images.

6) Compute homography matrix that warps one image to another.

7) Warp images and stitch them together to produce the panorama.

We used fine-grained timer to measure the computation time of each stage and identified filter convolution
and key point matching as the major performance bottleneck. The computation time of stitching two images
of size around 970 * 576 is as follow:

Compute Gaussian Pyramid: 30.32s

Compute DoG Pyramid: 0.05s

Detect Keypoints: 0.12s

Compute BRIEF Descriptor: 0.06s

Match keypoint descriptors: 55.66s

Compute Homography: 0.08s

Stitch Images: 0.34s


We believed that there is a lot of room for improvement and parallelism.
Currently the convolution is performed by iterating over each pixel in the image and computing
the weighted sum of the local neighborhood of the pixel.
Another way to do the convolution is to formulate the task as matrix multiplication.
We expect matrix multiplication to be more efficient but more memory consuming.

The key point matching uses the hamming distance between two descriptors as the distance metric. For each key point in
image 2, the algorithm finds its closet key point in image 1. Currently, the descriptor is represented as a vector of
integers, so comparing two descriptors needs to iterate over the vector. We can also represent the descriptor as a bit
array and computing hamming distance between integers should be faster.

Besides algorithmic optimization, the program will benefit greatly from parallelism. The steps up to computing brief
descriptors are independent for each image and can be done in parallel. Computing homography only considers every two
adjacent images and thus can be done in parallel for every pair of images.

## Issues

The program performs differently when the size of images varies. The results are more accurate when the images are
small. We observed significantly better results if we shrinked the input images. This is because there are
fewer key points and thus less noise in small images, the homography is usually more accurate. We can experiment with
different parameter configurations, such as the threshold to detect key point, in order to achieve better results with
large images.

## Platform Change

We decided to switch from latedays machine to GHC machines, and use CUDA to parallelize the program. We have experienced
several difficulties with compiling and linking the program with OpenCV on latedays. Fortunately, we successfully
compiled and ran our program on the GHC machine. Apart from the convenience of using OpenCV, we believed that the design
 of CUDA suits naturally with the image processing, especially when the parallelism is done in pixel level.

## Schedule

Despite some difficulties with the latedays platform, we are keeping up with the schedule and will be able to produce
reasonable quality panorama-like images and speed up the program using parallelizing techniques. We will probably not
have time to experiment with different descriptors, since we believe that the priority of this project is to maximize
the speedup of the parallel program.

Below is an updated schedule for the coming weeks

* **11.20--11.22**

Optimize the algorithm for filter convolution and key point matching. (Xin Xu)

* **11.23--11.26**

Start using CUDA to parallelize the program. (Zhuoqun Chen)

* **11.20--11.23**

Achieve a reasonable speedup with the parallel program. (Zhuoqun Chen)

* **11.24--11.26**

Iterate on the parallel version and optimize performance. (Xin Xu)

* **12.4--12.8**

Finish parallelizing the program and generate results. (Xin Xu, Zhuoqun Chen)

* **12.9-12.12**

Write final report. Prepare video and poster for demo. (Xin Xu, Zhuoqun Chen)


## Poster Session
We plan to demonstrate the process of processing a sequence of input images and results in the final panorama image
via a video. In addition, we will show the speedup graphs of our parallel algorithm against the sequential version.


## Resources

We follow the guidance from the following papers:

[1] Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua. BRIEF: Binary Robust Independent Elementary Features.

[2] Brown M,  Lowe DG. Automatic panoramic image stitching using invariant features, Int. J. Comput. Vis. , 2007, vol. 74 (pg. 59-73)


## Authors

* **Xin Xu** -  [xinx1](https://github.com/lotus-eater)
* **Zhuoqun Chen** -  [zhuoqunc](https://github.com/zq-chen)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
