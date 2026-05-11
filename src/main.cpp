#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    std::string modelPath = "C:/Users/Zzz/Documents/fad/models/face_detection_yunet_2023mar.onnx";
    // 1. Declare the pointer OUTSIDE the try block so it's accessible later
    cv::Ptr<cv::FaceDetectorYN> detector;

    try {
        detector = cv::FaceDetectorYN::create(modelPath, "", cv::Size(320, 320), 0.9f, 0.3f, 5000);
        if (detector.empty()) {
            std::cerr << "Error: Could not load model from " << modelPath << std::endl;
            return -1;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    detector->setInputSize(cv::Size(width, height));

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat faces;
        detector->detect(frame, faces);

        for (int i = 0; i < faces.rows; i++) {
            // Fix C4244 warnings by casting float to int
            int x = static_cast<int>(faces.at<float>(i, 0));
            int y = static_cast<int>(faces.at<float>(i, 1));
            int w = static_cast<int>(faces.at<float>(i, 2));
            int h = static_cast<int>(faces.at<float>(i, 3));

            cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Face Detection (YuNet)", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}