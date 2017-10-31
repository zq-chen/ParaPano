# ParaPano

Parallel image stitching using OpenMP.

## Summary

We are going to parallelize an image stitching program that aligns a set of images and stitch them together to produce a panorama. We will compare the speedup and quality of our parallel algorithm with the sequential version.

## Background

Automated panoramic image stitching is an interesting topic in computer vision. It consists of an interest point detector, a feature descriptor such as SIFT and BRIEF, and an image stitching tool based on feature matching. It extends the limitation of physical camera to capture scenes that cannot be captured in one frame, and easily generates the effects that would otherwise be expensive to produce. The program can be broken down into several highly parallelizable stages:

**1) Interest Point Detection**: interest points provide an efficient representation of the image. Interest points are found using Difference of Gaussian (DoG), which can be obtained by subtracting adjacent levels of a Gaussian Pyramid.

![alt text](https://user-images.githubusercontent.com/16803685/32247304-48e5ceee-be58-11e7-9880-db5f5eb824b0.png)

**2) Feature Descriptor**: feature descriptor characterizes the local information about an interest point. We will use either SIFT (Scale invariant feature descriptor) or BRIEF(Binary Robust Independent Elementary Features) as our choice of descriptor. If time allowed, we can experiment with different descriptors and compare the results.

**3) Matching Interest points**: match interest points using the distance of their descriptors.

**4) Align images**: compute the alignment of image pairs by estimating their homography (projection from one image to another).

**5) Stitching**: crop and blend the aligned images to produce the final result. If time allowed, we will address the problem of vertical “drifting” between image pairs by applying some “straightening” algorithm.

## Challenges

The performance bottleneck of the program lies in detecting the interest points, which involves convolving multiple filters and applying local neighborhood operations on the image. The highly expensive operation of processing local neighborhood of each pixel fits well with the data-parallel model, thus we believe that the program will benefit greatly from parallel implementation.

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

## Goals and Deiverables

We plan to complete the implementation of the parallel image stitching algorithm using BRIEF descriptor on Xeon Phi and compare the speedup against sequential version of the program. Apart from the performance, we also care about the quality of the resulting image after stitching. We plan to have reasonable quality panorama-like image after performing the algorithm.

If the project goes well, we hope to implement the algorithm with other descriptors like SIFT and ORB[3]. We hope to achieve better image stitching quality (rotation-variant and noise-resistant) with these descriptors or better speedup. If the work goes slowly, 

For demo, we plan to present our result by a video as well as speedup graphs. We plan to demonstrate the process of processing a sequence of input images and results in the final panorama image via a video. In addition, we will show the speedup graphs of our parallel algorithm against the sequential version.

[3] Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, November). ORB: An efficient alternative to SIFT or SURF. In Computer Vision (ICCV), 2011 IEEE international conference on(pp. 2564-2571). IEEE.

## Platform

We will use C++ and Xeon Phi machine, because the parallel programming interfaces we learned in class such as OpenMP and OpenMPI work well with C++ and Xeon Phi.



## Authors

* **Xin Xu** -  [PurpleBooth](https://github.com/lotus-eater)
* **Zhuoqun Chen** -  [PurpleBooth](https://github.com/zq-chen)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
