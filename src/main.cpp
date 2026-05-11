#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // This just checks if OpenCV is correctly linked
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    // Open the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Face Recognition Starter", frame);

        // Press 'q' to exit
        if (cv::waitKey(30) == 'q') break;
    }

    return 0;
}