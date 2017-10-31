# ParaPano

Parallel image stitching using OpenMP.

## Summary

We are going to parallelize an image stitching program that aligns a set of images and stitch them together to produce a panorama. We will compare the speedup and quality of our parallel algorithm with the sequential version.

## Background

Automated panoramic image stitching is an interesting topic in computer vision. It consists of an interest point detector, a feature descriptor such as SIFT and BRIEF, and an image stitching tool based on feature matching. It extends the limitation of physical camera to capture scenes that cannot be captured in one frame, and easily generates the effects that would otherwise be expensive to produce. The program can be broken down into several highly parallelizable stages:

<img src="https://user-images.githubusercontent.com/16803685/32248395-f8698b64-be5b-11e7-933c-25ecd84771af.png
" alt="img2" width="800" align="middle" />

**1) Interest Point Detection**: interest points provide an efficient representation of the image. Interest points are found using Difference of Gaussian (DoG), which can be obtained by subtracting adjacent levels of a Gaussian Pyramid.

<img src="https://user-images.githubusercontent.com/16803685/32247308-4e742ca2-be58-11e7-87ef-81cdaab4260b.png" alt="img2" width="800" align="middle" />

<img src="https://user-images.githubusercontent.com/16803685/32247312-523eccd4-be58-11e7-9b6c-e5fa2cc07e3a.png" alt="img3" width="800" align="middle" />


**2) Feature Descriptor**: feature descriptor characterizes the local information about an interest point. We will use either SIFT (Scale invariant feature descriptor) or BRIEF(Binary Robust Independent Elementary Features) as our choice of descriptor. If time allowed, we can experiment with different descriptors and compare the results.

**3) Matching Interest points**: match interest points using the distance of their descriptors.

<img src="https://user-images.githubusercontent.com/16803685/32247324-5722f748-be58-11e7-885f-cfc13b3831cb.png" alt="img4" width="800" align="middle" />

**4) Align images**: compute the alignment of image pairs by estimating their homography (projection from one image to another).

**5) Stitching**: crop and blend the aligned images to produce the final result. If time allowed, we will address the problem of vertical “drifting” between image pairs by applying some “straightening” algorithm.

<img src="https://user-images.githubusercontent.com/16803685/32247454-ce36ce90-be58-11e7-9cee-5a417001f309.png" alt="img5" width="800" align="middle" />

## Challenges

Building and parallelizing a panorama stitching program from scratch is a challenging work. The program is complex and consists of several stages that are dependent on each other. The stages in the process should be carried out sequentially, so after parallelizing each stage, we have to synchronize all the processors before proceeding to the next stage. This could incur large synchronization overhead. One challenge is to experiment with the trade-off between computation cost and synchronization cost.

To compute the interest points in each image, we need to generate a sequence of blurred versions of the original image, so when the size of the image or the number of images is large, the working set will inevitably not fit in cache. One challenge is to hide the memory access latency and reducing cache misses to achieve lower computation time.

From running the sequential version of the program in MATLAB, we believe that the performance bottleneck of the program lies in detecting the interest points, which involves convolving multiple filters and applying local neighborhood operations on the image. It is still a brute-force algorithm that searches every pixel as a potential interest point and thus incurs lots of computation. The highly expensive operation of processing local neighborhood of each pixel fits well with the data-parallel model. Therefore, we believe that the program will benefit greatly from parallel implementation.

### Dependency
The five stages in the pipeline are dependent on each other and should be carried out sequentially, yet there is a lot of parallelism in each stage, both within a single image and across multiple images.

### Memory Access
Processing pixels within an image exploits spatial locality.

### Communication
Communication happens when each image needs to match and stitch with its neighboring image. 


## Resources

We will build the program from scratch in C++. We have already implemented the sequential version of the algorithm (using BRIEF descriptor) in MATLAB.

We follow the guidance from the following papers:

[1] Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua. BRIEF: Binary Robust Independent Elementary Features.

[2] Brown M,  Lowe DG. Automatic panoramic image stitching using invariant features, Int. J. Comput. Vis. , 2007, vol. 74 (pg. 59-73)

[3] Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, November). ORB: An efficient alternative to SIFT or SURF. In Computer Vision (ICCV), 2011 IEEE international conference on(pp. 2564-2571). IEEE.

## Goals and Deiverables

We plan to complete the implementation of the parallel image stitching algorithm using BRIEF descriptor on Xeon Phi and compare the speedup against sequential version of the program. Apart from the performance, we also care about the quality of the resulting image after stitching. We plan to have reasonable quality panorama-like image after performing the algorithm.

If the project goes well, we hope to implement the algorithm with other descriptors like SIFT and ORB[3]. We hope to achieve better image stitching quality (rotation-variant and noise-resistant) with these descriptors or better speedup. If the work goes slowly, 

For demo, we plan to present our result by a video as well as speedup graphs. We plan to demonstrate the process of processing a sequence of input images and results in the final panorama image via a video. In addition, we will show the speedup graphs of our parallel algorithm against the sequential version.


## Platform

We will use C++ and Xeon Phi machine, because the parallel programming interfaces we learned in class such as OpenMP and OpenMPI work well with C++ and Xeon Phi.

We choose this parallel system because the workload of our problem needs high parallelism. Take interest point detection phase for instance, for each candidate pixel in the DoG, we need to compare its value with its surrounding neighboring pixels to determine whether it is local minimum/maximum. Given the large size of the image, this process can be significantly accelerated with parallelism which scales to hundreds of parallel working threads. In addition, the work of each parallel thread is similar, which could probably utilize the large (512-bit) vector width provided by Xeon Phi for further parallelism.

## Schedule

* **Week 1 11.1--11.5**

Understand the algorithm and the code in MATLAB. Start the sequential implementation in C++.

* **Week 2 11.6--11.12**

Finish the sequential version of the program and analyze the performance by fine-grained timing. Identify the bottleneck and come up with an approach to parallelize the program.

* **Week 3 11.13--11.19**

Start programming the parallel version of the program.

* **Week 4 11.20--11.26**

Iterate on the parallel version and optimize performance.

* **Week 5 11.27--12.3**

Finish parallelizing the program and generate results. If there is time left, finish the additional goals of the project.

* **Week 6 12.4--12.12**

Wrap up the project. Write final report. Prepare video and poster for demo.


## Authors

* **Xin Xu** -  [xinx1](https://github.com/lotus-eater)
* **Zhuoqun Chen** -  [zhuoqunc](https://github.com/zq-chen)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

