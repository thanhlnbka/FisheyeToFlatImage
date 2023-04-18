#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include<string>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>


using namespace cv;


static float faceTransform[6][2] =
        {
                // left front right back top bottom
                {M_PI / 2, 0},      // right    +x
                {-M_PI / 2, 0},     // left     -x
                {0, -M_PI / 2},     // top      +y
                {0, M_PI / 2},      // bottom   -y
                {0, 0},             // front    +z
                {M_PI, 0},          // back     -z
        };



inline void createCubeMapFace(const cv::Mat &in, cv::Mat &face, int faceId = 0, int width = -1, int height=-1){
    float inWidth = in.cols;
    float inHeight = in.rows;

    if (width < 0) {
        width = inWidth / 4;
        height = width;
    }
    // Allocate map
    cv::Mat mapx(height, width, CV_32F);
    cv::Mat mapy(height, width, CV_32F);
    const float an = sin(M_PI / 4);
    const float ak = cos(M_PI / 4);

    const float ftu = faceTransform[faceId][0];
    const float ftv = faceTransform[faceId][1];
    // calculate the corresponding source coordinates.
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {

            // Map face pixel coordinates to [-1, 1] on plane
            float nx = (float)y / (float)height - 0.5f;
            float ny = (float)x / (float)width - 0.5f;

            nx *= 2;
            ny *= 2;

            nx *= an;
            ny *= an;

            float u, v;

            // Project from plane to sphere surface.
            if(ftv == 0) {
                // Center faces
                u = atan2(nx, ak);
                v = atan2(ny * cos(u), ak);
                u += ftu;
            } else if(ftv > 0) {
                // Bottom face
                float d = sqrt(nx * nx + ny * ny);
                v = M_PI / 2 - atan2(d, ak);
                u = atan2(ny, nx);
            } else {
                // Top face
                float d = sqrt(nx * nx + ny * ny);
                v = -M_PI / 2 + atan2(d, ak);
                u = atan2(-ny, nx);
            }

            // Map from angular coordinates to [-1, 1], respectively.
            u = u / (M_PI);
            v = v / (M_PI / 2);

            // Warp around, if our coordinates are out of bounds.
            while (v < -1) {
                v += 2;
                u += 1;
            }
            while (v > 1) {
                v -= 2;
                u += 1;
            }

            while(u < -1) {
                u += 2;
            }
            while(u > 1) {
                u -= 2;
            }

            // Map from [-1, 1] to in texture space
            u = u / 2.0f + 0.5f;
            v = v / 2.0f + 0.5f;

            u = u * (inWidth - 1);
            v = v * (inHeight - 1);

            // Save the result for this pixel in map
            mapx.at<float>(x, y) = u;
            mapy.at<float>(x, y) = v;
        }
    }

    // Recreate output image if it has wrong size or type.
    if(face.cols != width || face.rows != height ||
       face.type() != in.type()) {
        face = cv::Mat(width, height, in.type());
    }
    // Do actual resampling using OpenCV's remap
    remap(in, face, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (faceId == 2) {
        // top, rotation clockwise
        cv::rotate(face, face, cv::ROTATE_90_CLOCKWISE );
    } else if (faceId == 3) {
        // bottom, rotation anti-clockwise
        cv::rotate(face, face, cv::ROTATE_90_COUNTERCLOCKWISE );
    }
}


Mat convert_to_panorama(Mat img, int number_block){
    int l = img.cols / 2;
    int vote_pixel = number_block;
    Mat destination = Mat(l, vote_pixel * l, CV_8UC3);
    int i, j;
    int x, y;
    double radius, theta;
    double fTrueX, fTrueY;
    for (i = 0; i < destination.rows; ++i)
        {
            for (j = 0; j < destination.cols; ++j)
            {
                radius = (double)(l - i);
                theta = 2.0 * M_PI * (double)(vote_pixel * l - j) / (double)(vote_pixel * l);

                fTrueX = radius * cos(theta);
                fTrueY = radius * sin(theta);

                x = (int)(round(fTrueX)) + l;
                y = l - (int)(round(fTrueY));
                // check bounds
                if (x >= 0 && x < (2 * l) && y >= 0 && y < (2 * l))
                {
                    destination.at<Vec3b>(i, j) = img.at<Vec3b>(y, x);
                }
            }
        }

    return destination;
}



int main(int argc, char){
    
    auto start_time = std::chrono::high_resolution_clock::now();
    Mat img = imread("../images/source.jpg");
    // Check if the frame is empty
    if (img.empty()) {
        std::cout << "Image is empty !" << std::endl;
    }
    // Convert fisheye to panorama image 
    img = convert_to_panorama(img, 4);
    int width = img.cols; 
    int height = img.rows;
    // Crop image and normalize bottom face
    // Define the ROI rectangle
    int x1 = 0;
    int y1 = 0;
    int x2 = width;
    int y2 = 3*height/4; // Remove bottom from panorama image

    Rect roi(x1, y1, x2 - x1, y2 - y1);
    // Crop the image using the ROI
    Mat croppedImage = img(roi);
    Mat image_bottom;
    createCubeMapFace(img, image_bottom, 3, -1, -1);
    
    imwrite("../outtests/panorama.jpg", croppedImage); 
    imwrite("../outtests/panorama_bottom.jpg", image_bottom);

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken by code: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}