# ParaPano

Parallel image stitching using CUDA.

## Summary

We parallelize an auto image stitching program using CUDA that combines a set of images and stitches them together to produce a panorama. Using various parallelizing techniques and algorithmic optimizations, we achieve multi-thousand speedup compared with the sequential version of the program. For deliverables, we will show a demo of multi-photo stitching using ParaPano to produce high-quality panorama.

See <a href="https://zq-chen.github.io/ParaPano/Proposal">Project proposal</a>, and <a href="https://zq-chen.github.io/ParaPano/Milestone">Milestone report</a>.

## Background and Algorithm

Automated panoramic image stitching is an interesting topic in computer vision. It consists of an interest point detector, a feature descriptor and an image stitching tool based on feature matching. It extends the limitation of physical camera to capture scenes that cannot be captured in one frame, and easily generates the effects that would otherwise be expensive to produce. 

The entire algorithm can be broken down into several parallelizable stages, which are introduced in the following sections in detail. The two input images for stitching used in our algorithm introduction are shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/32248395-f8698b64-be5b-11e7-933c-25ecd84771af.png" alt="img1" width="800" align="middle" />

### Key Point Detection

Key points provide an efficient representation of the image. To find the key points, we convolve a set of Gaussian filters on the image to produce a Gaussian Pyramid. The Gaussian pyramid of the example input image is shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/32247308-4e742ca2-be58-11e7-87ef-81cdaab4260b.png" alt="img2" width="800" align="middle" />

Then we obtain the Difference of Gaussian (DoG) by subtracting adjacent levels of the Gaussian Pyramid.  Points whose DoG values satisfy certain criteria are considered as key points. The difference of Gaussian and the detected key points for the example image is shown in the following two figures.

<img src="https://user-images.githubusercontent.com/16803685/33908392-40b9f9e2-df56-11e7-9e7e-5ab0f2340a51.jpg" alt="img3" width="800" align="middle" />

<img src="https://user-images.githubusercontent.com/16803685/33950627-e39717c2-dffa-11e7-8561-30c4e6384432.png" alt="img4" width="800" align="middle" />


### Feature Descriptor

Feature descriptor characterizes the local information about an interest point. We used BRIEF (Binary Robust Independent Elementary Features) as our choice of descriptor. 

BRIEF considers an image patch around a key point, selects a set of location pairs in the image patch, and compares the intensity values of these location pairs. For example, X and Y are two locations, p(X) and p(Y) represent their intensity values, then the comparison result is 1 if p(X) < p(Y), 0 otherwise.

We randomly sample 256 pairs of locations in a 9 x 9 grid using a Gaussian distribution centered around the origin, and freeze this set to calculate the BRIEF descriptors. In this way, each key point is described using 256 values, each value specifying the comparison result of one of the 256 pairs. The following figure shows the visualization of sampled locations, where each line connects a location pair.

<img src="https://user-images.githubusercontent.com/16803685/33913048-37355352-df66-11e7-9c47-6d1d0557dcc0.jpg" alt="img5" width="500" align="middle" />

### Matching Key Points

We then match key points using the distance of their BRIEF descriptors. Hamming distance, i.e. the number of positions where two descriptors differ, is used as the distance metric. Given the descriptors of two images, we find the best matching point in the second image for each key point in the first image. The pseudo code for matching key point is shown as follows.

```c++
for p1 in image1.keypoints:
    float min = MAX_FLOAT;
    float second_min = MAX_FLOAT;
    Keypoint best_match = NULL;
    for p2 in image2.keypoints:
        float d = dist(p1.descriptor, p2.descriptor);
        if d < min
           second_min = min;
           min = d;
           best_match = p2;
        else if d < second_min
           second_min = d;
    if min / second_min < 0.8
    	add (p1, best_match) to matching key point pairs
```


We find both the minimum and the second minimum distance between the descriptor to be matched and some other descriptor in the second image. To reduce the likelihood of incorrect matching, we only consider the matching as valid if the difference between the minimum distance and second minimum distance is large enough. The matched key points between Fig 1(a) and Fig 1(b) are shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/32247324-5722f748-be58-11e7-885f-cfc13b3831cb.png" alt="img6" width="800" align="middle" />

### Compute Homography between Image Pairs

Compute the homography, a $3 \times 3$ transformation matrix that warps one image onto another, between each adjacent images. This is done by keep randomly sampling four pairs of matching key points of two images and computing an estimated homography until a good homography is found.

### Warp and Blend Images

Finally, we stitch the images by warping and blending the images onto an common plane to produce the panorama. The stitching result for the example input images is shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/32247454-ce36ce90-be58-11e7-9cee-5a417001f309.png" alt="img7" width="700" align="middle" />

### Workload Analysis

We use fine-grained timer to measure the computation time of each stage and identify the filter convolution and key point matching as the major performance bottlenecks. The detailed benchmark performance is elaborated in Approaches Section.

The five stages in the pipeline are dependent on each other and should be carried out sequentially, yet there is a lot of parallelism in each stage, both within a single image and across multiple images. Filter convolution considers the local neighborhood centered around each pixel, and thus exploiting strong spatial locality. Processing the image involves performing the same operations on each pixel, so it fits well with the data-parallel model. Therefore, the program can benefit greatly from parallel implementation.


## Approaches

### Technologies and Target Machines

We use C++ and CUDA for programming. In specific, our application is targeted for machine with Nvidia GPU. We develop our application on machines with Nvidia GeForce GTX 1080, which supports CUDA compute capability 6.1.


### Algorithm Parallelization and Optimization

In this section, we elaborate our process of parallelizing and optimizing several parts of the image-stitching algorithm.

We first implement a sequential CPU version (using single thread) from scratch as a baseline (See Milestone Report). We benchmark the running time for each section of the algorithm on stitching two images of size of 970 * 576 and obtain the performance summary for our sequential version in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/33913045-3712e6e6-df66-11e7-9eaf-5f56368e8905.png" alt="img8" width="600" align="middle" />

As can be easily observed, for stitching two images, computing Gaussian pyramid and matching key point are the two parts of the algorithm that consumes the most amount of time. We believed that there is a lot of room for improvement and parallelism.


### Gaussian Pyramid Computation

The convolution in Gaussian pyramid computation was performed by iterating over each pixel in the image and computing the weighted sum of the local neighborhood of the pixel. This computation fits naturally with the data-parallel model of CUDA, so we choose to parallelize within each image. We spawn a threaded-block with the size of the image, and each thread in the block performs the convolution for a single pixel. We synchronize the CUDA threads at the end of the computation of Gaussian Pyramid.

### Key Point Matching

#### Algorithmic Optimization of Key Point Matching

The key point matching process uses the hamming distance between two BRIEF descriptors as the distance metric. For each key point in image 2, the algorithm finds its closest key point in image 1. In our baseline sequential implementation, a BRIEF descriptor is implemented as a vector of 256 integers with 0/1 values. Therefore, comparing two descriptors needs to iterate over the vector, which is very time-consuming.

```c++
struct Descriptor {
   int values[NUM_OF_TEST_PAIRS];
};
```

Our  first decision is to improve the computation time for hamming distance by replacing the data structure  that represents the BRIEF descriptor with a bit array, i.e. std::bitset in C++. In this way, computing hamming distance between two descriptors is essentially XORing the two bit arrays and counting the number of set bits of that bit array. Counting the number of set bits on a bit array is simply calling std::bitset::count function.

```c++
struct Descriptor {
   std::bitset<NUM_OF_TEST_PAIRS> values;
};
```

We first test the performance of using bit array as descriptor on sequential CPU program for 2-image stitching. The computation time for matching key points decreases significantly: from 49.55 seconds to only 1.34 seconds. This huge improvement is mainly due to the efficiency provided by std::bitset, whose bit operations are finely optimized in C++.

However, the matching speed performance using bit array on CPU gets worse when stitching more than 2 images. The key point matching process takes 30.01 seconds for 3-image stitching, and 87.32 seconds for 4-image stitching. The matching time for 5-image stitching is even more unbearable.  We find that as we add more images for stitching, the number of total key points increases significantly. Sequential key point matching gets inevitably slower with increasing workload. Parallelism is needed.


#### Parallelization of Key Point Matching

We decide to parallelize the key point matching process using CUDA. However, it is not possible to simply utilize std::bitset in CUDA device code since bitset is not implemented in CUDA. Thus we implement our own version of bitset in CUDA. We use 4 64-bit unsigned integers to represent the 256-bit descriptor, and use bit-manipulation to set and count the bits in the descriptor efficiently.

```c++
struct Descriptor {
  uint64_t num0;
  uint64_t num1;
  uint64_t num2;
  uint64_t num3;
};
```

To match key points between two images in parallel, our approach is to parallelize over the all the key points of the first image. For each key point in the first image, a CUDA thread is spawned. Each thread iterates over all the key points in the second image, and compute the hamming distances between these key points and the key point in the first image to select the closest matching key point.

Specifically we find that the algorithm to count the number of set bits during hamming distance computation has a great impact on the running time, since bit counting is performed tens of thousands of time on each CUDA thread. Linear-time approach, i.e. iterating over descriptor elements, has already shown to be too slow in CPU version benchmark. We thus compare the performance of two approaches: $O(k)$ algorithm where k is the number of set bits, and constant-time algorithm using bit tricks on 64-bit integers [3]. We find that the constant-time algorithm outperforms the O(k) algorithm by a large margin. Detailed performance evaluation results are presented in Performance Evaluation Section. 

### Algorithmic Optimization

We notice that our program does not produce the correct result when the number of images is larger than 4. After further investigation, we realize that because we warp all the images relative to the first image, the later images become more distorted, and they eventually become too large to fit into memory. To fix this problem, we adjust the warping of each image to be relative to the center image. This is done by multiplying the homography of each image by the inverse of the homography of the center image. The following figure shows the output before and after adjusting the warping.

<img src="https://user-images.githubusercontent.com/16803685/33950677-033e1d8c-dffb-11e7-9b35-1322994c7439.png" alt="img9" width="800" align="middle" />


## Performance Evaluation

### Metrics

As our goal is to optimize the sequential algorithm to run faster with GPU acceleration, we use computation time of the algorithm, as well as the speedup with respect to baseline sequential CPU version as two main metrics for performance.

### Experimental Setup

We test our program on various sets of images and obtain successful results. To measure the speedup of our program, we evaluate the performance on the two image sets of different sizes below. Image Set II has more images, larger image size and much more key points than Image Set I, so it is more computationally difficult and intensive.

#### Image Set I

Image set I contains 2 images. The size of the images are 945 x 576 for Image 1, and 1064 x 576 for Image 2. The two images are shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/33950733-242ff59c-dffb-11e7-91e8-eb53f241de97.png" alt="img10" width="800" align="middle" />

The number of key points detected for Image 1 is 7814, and 4767 for Image 2.

### Image Set II

Image Set II contains 5 images taken with iPhone. 
The size of each image is 1440 x 1080. The five images are shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/33950754-3089ece4-dffb-11e7-9fbc-c645a8d82a80.png" alt="img11" width="800" align="middle" />

The number of key points detected for each of the five images are shown in the following table.

| Image         |  Number of key points detected |
| ------------- |:------------------------------:|
| Image 1       | 9333                           |
| Image 2       | 17198                          | 
| Image 3       | 35325                          |
| Image 4       | 40924                          |
| Image 5       | 54025                          |


### Quality of Panorama Results

In this section, we demonstrate the panorama results generated by ParaPano.

The panorama result for Image Set I is shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/33953364-bb6406a4-e002-11e7-9ef2-74aed1080e0b.jpeg" alt="img12" width="800" align="middle" />

The panorama result for Image Set II is shown in Figure 

<img src="https://user-images.githubusercontent.com/16803685/33913014-14f9225a-df66-11e7-94a1-5e74fb3abced.jpg" alt="img13" width="800" align="middle" />

The panorama results are both of convincing quality. All images from the image set are warped relative to the center and stitched together smoothly. The boundaries of the adjacent images are blended nicely with each other. See Appendix for more panorama results generated by ParaPano.

### Results of Performance Optimization

In this section, we present the performance optimization results for our parallelized algorithms.

#### Results of Parallelizing Gaussian Filtering

The process of Gaussian filtering benefits greatly from parallelism. The computation time is reduced to be within second. We are able to achieve 505.3x speedup on Image Set I, 346.4x speedup on the first two images of Image Set II,  367.6x on the first three images, and 367.4x on the first four images. The results are summarized in the following two figures.

<img src="https://user-images.githubusercontent.com/16803685/33952141-ef10f79a-dffe-11e7-993b-f3a5908b68a9.png" alt="img14" width="600" align="middle" />

<img src="https://user-images.githubusercontent.com/16803685/33913046-371c4100-df66-11e7-91ac-0ed4f58c8406.png" alt="img15" width="600" align="middle" />

Because we parallelize the computation within each image, the scale of parallelism increases as the number of pixels in the image increases, we expect the speedup to scale with the average size of the input images. However we notice that the speedup is greater for Image Set I which has smaller images. We reason that this is due to memory transfer overhead between the CUDA host and device, and synchronization overhead of CUDA threads. Because Image Set I has only two images of smaller size, much less data is transferred between the host and the device. There are also fewer number of threads spawned, so there is less synchronization overhead after each Gaussian filtering. To confirm our conjecture, we perform a fine-grain time measure on the parallelized computation of Gaussian filtering on Image Set II. We find out that although the actual computation on CUDA device is only 0.43s, transferring the images to CUDA device takes 0.73s, which is almost twice as much as the actual computation time.

#### Results of Parallelizing Key Point Matching

We compare the computation time of key point matching on four versions of algorithms: baseline CPU sequential version, CPU sequential using bitset descriptors, CUDA with boolean array descriptors and CUDA with self-implemented bitset descriptors. The computation time of key point matching is shown in the following figure.

<img src="https://user-images.githubusercontent.com/16803685/33913003-0aca750e-df66-11e7-938d-61fc6f93beac.png" alt="img16" width="600" align="middle" />

We obtain a significant speedup on CUDA with self-implemented bitset. The speedup of key point matching with respect to the baseline CPU sequential version is shown in Figure ~\ref{Fig17}. Compared to the sequential baseline, we achieve 4955 speedup on Image Set I, and 17227.85x speedup on Image Set II. The parallelized key point matching algorithm scales with the number of key points detected in the input image. Because we compress the descriptor structure from 256 32-bit integers to 4 64-bit integers, we significantly reduce the amount of memory transferred between CUDA host and device. Given that there are tens of thousands of key points in each image, this optimization greatly reduces the memory overhead.

<img src="https://user-images.githubusercontent.com/16803685/33913047-37260e4c-df66-11e7-8348-4511a6870c22.png" alt="img17" width="600" align="middle" />

Finally, we also compare two algorithms that count set bits of a bit array in CUDA. The choice of this algorithm has great impact on performance since hamming distances computation is carried out on each CUDA thread tens of thousands of time for key point matching. We first experiment with the Brian Kernighan's algorithm, which performs n & (n-1) in a loop. This algorithm takes O(k) where k is the number of set bits.

```c++
int count_set_bit(int n) {
    int count = 0;
    while (n)
    {
        n &= (n-1) ;
        count++;
    }
    return count;
}
```

Although this algorithm is relatively efficient, given the amount of computation, it still does not achieve the speedup that we aim for. After some research [3], we learn about the bitwise-SWAR (SIMD Within A Register) algorithm which exploits parallelism in hardware.

```c++
int constant_count_set_bit(int n) {
    i = i - ((i >> 1) & 0x5555555555555555);
    i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
```

This method speeds up the computation by performing SIMD parallel operation on multiple elements in the vector at the same time.

### Performance Limitation

After optimizing Gaussian filter computation and key point matching, the entire algorithm takes around 10 seconds to complete multi-large-image stitching, with most of the stages taking at most 1 second. However, more than half of the running time is spent on the final stage: warping and blending the images to produce the panorama image. This is the only bottleneck left in the algorithm, which can be observed from the breakdown of the computation time on Image Set II in the figure in the next section.



We find that this final stage, especially image blending, is the most difficult part of the algorithm to parallelize. The difficulty lies in the fact that warped images are blended onto a common plane, where adjacent images are dependent on each other. Without significant modification of the blending algorithm, this process needs to be carried out sequentially.

In order to parallelize the image warping, we utilize the OpenCV CUDA module's warpPerspective library function. This provides a speedup of 2.4 compared with sequential warping, as shown in the following table.

| Algorithm           |  Warp and Blend Time (second)  |
| ------------------- |:------------------------------:|
| Sequential Warping  | 4.40                           |
| CUDA Warping        | 1.84                           | 


As for image blending, we still have to make it run sequentially since we haven't been able to come up with a parallel algorithm to decouple the dependencies between the images.

### Execution Time Breakdown

The execution time breakdown for ParaPano running on Image Set II is shown in the following figure. The total time is 5.63 seconds, which is within reasonable range for stitching five large photos.

<img src="https://user-images.githubusercontent.com/16803685/33952687-99e3658a-e000-11e7-8eb2-15b2f532015c.png" alt="img18" width="700" align="middle" />

We can observe that of all the stages, warping and blending images consumes the most amount of time (23.3%), even though we have used CUDA warp in OpenCV to parallelize warping. This bottleneck is due to the difficulty to parallelize image blending, as has been discussed in Section 4.5. 

The running time improvement of Gaussian pyramid computation is shown in detail in previous section. Note that for parallelized Gaussian pyramid computation, there's a non-trivially long time (0.73 second) caused by CUDA memory transfer.
The optimization detail for key point matching process is elaborated in previous section. 

For other stages of the algorithm, the room for improvement is relatively small. These stages, such as I/O (reading input images and writing the out panorama image) and homography computation etc,  can gain little benefit from CUDA parallelism because of the low level of parallelism and potentially high I/O overhead on CUDA. 

Computing homography takes 14.9% of the total execution time.
To compute the homography which is a 3 x 3 matrix, we need to randomly sample 4 pairs of matching key points, construct a set of linear equations, and perform a SVD decomposition to find an estimated homography. We test the estimated homography by calculating its error on all matching key points. If the error is too large, we restart this process again.

```c++
while (error(H) > threshold)
    sample 4 pairs of matching key points
    solve Ax=b to find H
return H
```

This process of trial-and-error does appear to be easily parallelizable. The computation within each loop takes little time, so parallelism needs to be done across the while loops. However, there are too many possible combinations of four pairs of matching key points of two images, we cannot parallelize the computation on all of these combinations. 

In summary, except from implementing a new parallel blending algorithm, there's little room left for execution time improvement.

### Choice of Machine

The workload of the image stitching task is very suitable for GPU machines. It is composed of several stages which have the potential to be highly parallelized. GPU machines are ideal for this task, because the most time-consuming stages in sequential version (Gaussian filter computation and key point matching) possess forms of data parallelism, at both image-level and pixel-level. Due to the large problem size, the parallel algorithm can easily scale to tens of thousands of threads, and GPU is the best platform that provides this high level of parallelism.



## References

We follow the guidance from the following papers:

[1] Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua. BRIEF: Binary Robust Independent Elementary Features.

[2] Brown M,  Lowe DG. Automatic panoramic image stitching using invariant features, Int. J. Comput. Vis. , 2007, vol. 74 (pg. 59-73)

[3] Efficient Hamming Distance implementations. https://en.wikipedia.org/wiki/Hamming_weight


## Appendix

More panorama results generated by ParaPano are presented in this section.

<img src="https://user-images.githubusercontent.com/16803685/33950792-4759f43c-dffb-11e7-9a23-eeef5baf10e8.png" alt="img19" width="800" align="middle" />


<img src="https://user-images.githubusercontent.com/16803685/33908381-389f4ee2-df56-11e7-8f64-2ccf3a38080d.jpg" alt="img20" width="800" align="middle" />




## Authors

* **Xin Xu** -  [xinx1](https://github.com/lotus-eater)
* **Zhuoqun Chen** -  [zhuoqunc](https://github.com/zq-chen)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
