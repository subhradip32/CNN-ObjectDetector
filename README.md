# CNN OBJECT DETECTOR
This project is based on YOLOV8 object detection this application uses the power of streamlit applications to build the fornt end powered by the powerfull Computervision technology by Opencv. This is a very basic application which can be ussed to object detected by using the yolov8.

# YOLOv8
Introducing Ultralytics YOLOv8, the latest version of the acclaimed real-time object detection and image segmentation model. YOLOv8 is built on cutting-edge advancements in deep learning and computer vision, offering unparalleled performance in terms of speed and accuracy. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs.

Explore the YOLOv8 Docs, a comprehensive resource designed to help you understand and utilize its features and capabilities. Whether you are a seasoned machine learning practitioner or new to the field, this hub aims to maximize YOLOv8's potential in your projects
Where to Start
Install ultralytics with pip and get up and running in minutes    Get Started
Predict new images and videos with YOLOv8    Predict on Images
Train a new YOLOv8 model on your own custom dataset    Train a Model
Tasks YOLOv8 tasks like segment, classify, pose and track    Explore Tasks
NEW 🚀 Explore datasets with advanced semantic and SQL search    Explore a Dataset

YOLO: A Brief History
YOLO (You Only Look Once), a popular object detection and image segmentation model, was developed by Joseph Redmon and Ali Farhadi at the University of Washington. Launched in 2015, YOLO quickly gained popularity for its high speed and accuracy.

1. YOLOv2: released in 2016, improved the original model by incorporating batch normalization, anchor boxes, and dimension clusters.
2. YOLOv3: launched in 2018, further enhanced the model's performance using a more efficient backbone network, multiple anchors and spatial pyramid pooling.
3. YOLOv4: was released in 2020, introducing innovations like Mosaic data augmentation, a new anchor-free detection head, and a new loss function.
4. YOLOv5: further improved the model's performance and added new features such as hyperparameter optimization, integrated experiment tracking and automatic export to popular export formats.
5. YOLOv6: was open-sourced by Meituan in 2022 and is in use in many of the company's autonomous delivery robots.
6. YOLOv7: added additional tasks such as pose estimation on the COCO keypoints dataset.
7. YOLOv8: is the latest version of YOLO by Ultralytics. As a cutting-edge, state-of-the-art (SOTA) model, YOLOv8 builds on the success of previous versions, introducing new features and improvements for enhanced performance, flexibility, and efficiency. YOLOv8 supports a full range of vision AI tasks, including detection, segmentation, pose estimation, tracking, and classification. This versatility allows users to leverage YOLOv8's capabilities across diverse applications and domains.
8. YOLOv9: Introduces innovative methods like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN).

# Opencv
OpenCV (Open Source Computer Vision Library: http://opencv.org) is an open-source library that includes several hundreds of computer vision algorithms. The document describes the so-called OpenCV 2.x API, which is essentially a C++ API, as opposed to the C-based OpenCV 1.x API (C API is deprecated and not tested with "C" compiler since OpenCV 2.4 releases)

OpenCV has a modular structure, which means that the package includes several shared or static libraries. The following modules are available:

Core functionality (core) - a compact module defining basic data structures, including the dense multi-dimensional array Mat and basic functions used by all other modules.
Image Processing (imgproc) - an image processing module that includes linear and non-linear image filtering, geometrical image transformations (resize, affine and perspective warping, generic table-based remapping), color space conversion, histograms, and so on.
Video Analysis (video) - a video analysis module that includes motion estimation, background subtraction, and object tracking algorithms.
Camera Calibration and 3D Reconstruction (calib3d) - basic multiple-view geometry algorithms, single and stereo camera calibration, object pose estimation, stereo correspondence algorithms, and elements of 3D reconstruction.
2D Features Framework (features2d) - salient feature detectors, descriptors, and descriptor matchers.
Object Detection (objdetect) - detection of objects and instances of the predefined classes (for example, faces, eyes, mugs, people, cars, and so on).
High-level GUI (highgui) - an easy-to-use interface to simple UI capabilities.
Video I/O (videoio) - an easy-to-use interface to video capturing and video codecs.
... some other helper modules, such as FLANN and Google test wrappers, Python bindings, and others.
The further chapters of the document describe functionality of each module. But first, make sure to get familiar with the common API concepts used thoroughly in the library.

API Concepts
cv Namespace
All the OpenCV classes and functions are placed into the cv namespace. Therefore, to access this functionality from your code, use the cv:: specifier or using namespace cv; directive:
```
#include "opencv2/core.hpp"
...
cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 5);
...
or :

#include "opencv2/core.hpp"
using namespace cv;
...
Mat H = findHomography(points1, points2, RANSAC, 5 );
```
Some of the current or future OpenCV external names may conflict with STL or other libraries. In this case, use explicit namespace specifiers to resolve the name conflicts:
```
Mat a(100, 100, CV_32F);
randu(a, Scalar::all(1), Scalar::all(std::rand()));
cv::log(a, a);
a /= std::log(2.);
```
Automatic Memory Management
OpenCV handles all the memory automatically.

First of all, std::vector, cv::Mat, and other data structures used by the functions and methods have destructors that deallocate the underlying memory buffers when needed. This means that the destructors do not always deallocate the buffers as in case of Mat. They take into account possible data sharing. A destructor decrements the reference counter associated with the matrix data buffer. The buffer is deallocated if and only if the reference counter reaches zero, that is, when no other structures refer to the same buffer. Similarly, when a Mat instance is copied, no actual data is really copied. Instead, the reference counter is incremented to memorize that there is another owner of the same data. There is also the cv::Mat::clone method that creates a full copy of the matrix data. See the example below:
```
// create a big 8Mb matrix
Mat A(1000, 1000, CV_64F);
 
// create another header for the same matrix;
// this is an instant operation, regardless of the matrix size.
Mat B = A;
// create another header for the 3-rd row of A; no data is copied either
Mat C = B.row(3);
// now create a separate copy of the matrix
Mat D = B.clone();
// copy the 5-th row of B to C, that is, copy the 5-th row of A
// to the 3-rd row of A.
B.row(5).copyTo(C);
// now let A and D share the data; after that the modified version
// of A is still referenced by B and C.
A = D;
// now make B an empty matrix (which references no memory buffers),
// but the modified version of A will still be referenced by C,
// despite that C is just a single row of the original A
B.release();
 
// finally, make a full copy of C. As a result, the big modified
// matrix will be deallocated, since it is not referenced by anyone
C = C.clone();
```
You see that the use of Mat and other basic structures is simple. But what about high-level classes or even user data types created without taking automatic memory management into account? For them, OpenCV offers the cv::Ptr template class that is similar to std::shared_ptr from C++11. So, instead of using plain pointers:
```
T* ptr = new T(...);
you can use:

Ptr<T> ptr(new T(...));
or:

Ptr<T> ptr = makePtr<T>(...);
Ptr<T> encapsulates a pointer to a T instance and a reference counter associated with the pointer. See the cv::Ptr description for details.
```
Automatic Allocation of the Output Data
OpenCV deallocates the memory automatically, as well as automatically allocates the memory for output function parameters most of the time. So, if a function has one or more input arrays (cv::Mat instances) and some output arrays, the output arrays are automatically allocated or reallocated. The size and type of the output arrays are determined from the size and type of input arrays. If needed, the functions take extra parameters that help to figure out the output array properties.

Example:
```
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
 
using namespace cv;
 
int main(int, char**)
{
 VideoCapture cap(0);
 if(!cap.isOpened()) return -1;
 
 Mat frame, edges;
 namedWindow("edges", WINDOW_AUTOSIZE);
 for(;;)
 {
 cap >> frame;
 cvtColor(frame, edges, COLOR_BGR2GRAY);
 GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
 Canny(edges, edges, 0, 30, 3);
 imshow("edges", edges);
 if(waitKey(30) >= 0) break;
 }
 return 0;
}
```

# Reesult 
![Screenshot from 2024-05-13 23-16-25](https://github.com/subhradip32/CNN-ObjectDetector/assets/83198378/170fe41f-c2d2-407d-9f7b-e98a0f39ad31)
This is the view of the freshly launched application.
![Screenshot from 2024-05-13 23-16-47](https://github.com/subhradip32/CNN-ObjectDetector/assets/83198378/e6d773fc-5595-48bc-8b22-7a0cbe27b140)
By selecting the image from your local pc and uploaded on to the application it can be used for detection. 
![Screenshot from 2024-05-13 23-17-01](https://github.com/subhradip32/CNN-ObjectDetector/assets/83198378/6ecb583c-c1f5-4fb0-8a47-9d3b94b21b73)
Now by clicking the download button we can download the image in BGR format. 
