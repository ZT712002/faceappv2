#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <iostream>

using namespace dlib;
using namespace std;


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;



int main() {
    try {
        // 1. Load YuNet
        string yunetPath = "C:/Users/Zzz/Documents/fad/models/face_detection_yunet_2023mar.onnx";
        cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(yunetPath, "", cv::Size(320, 320));
        if (detector.empty()) throw std::runtime_error("Failed to load YuNet model.");

        // 2. Load Dlib Models (IMPORTANT: Extract the .bz2 files first!)
        shape_predictor sp;
        // Point to the extracted .dat files, NOT the .bz2 files
        deserialize("C:/Users/Zzz/Documents/fad/models/shape_predictor_5_face_landmarks.dat") >> sp;
        
        anet_type net;
        deserialize("C:/Users/Zzz/Documents/fad/models/dlib_face_recognition_resnet_model_v1.dat") >> net;

        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        detector->setInputSize(cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        cv::Mat frame;
        int frame_count = 0;
        
        // A structure to hold face data between heavy Dlib recognition frames
        struct TrackedFace {
            cv::Rect box;
            string label = "Processing...";
        };
        std::vector<TrackedFace> tracked_faces;

        while (cap.read(frame)) {
            frame_count++;
            cv::Mat faces;
            detector->detect(frame, faces);

            // 1. If it's a "heavy" frame, update both detection AND recognition
            if (frame_count % 6 == 0) { 
                tracked_faces.clear();
                cv_image<bgr_pixel> dlib_img(frame);

                for (int i = 0; i < faces.rows; i++) {
                    int x = static_cast<int>(faces.at<float>(i, 0));
                    int y = static_cast<int>(faces.at<float>(i, 1));
                    int w = static_cast<int>(faces.at<float>(i, 2));
                    int h = static_cast<int>(faces.at<float>(i, 3));

                    dlib::rectangle d_rect(x, y, x + w, y + h);
                    if (d_rect.is_empty()) continue;

                    // Heavy operations: landmarks + ResNet
                    auto shape = sp(dlib_img, d_rect);
                    matrix<rgb_pixel> face_chip;
                    extract_image_chip(dlib_img, get_face_chip_details(shape, 150, 0.25), face_chip);
                    
                    // The massive bottleneck line:
                    matrix<float, 0, 1> face_descriptor = net(face_chip); 

                    TrackedFace tf;
                    tf.box = cv::Rect(x, y, w, h);
                    tf.label = "Face Encoded"; // Here you would normally compare descriptors
                    tracked_faces.push_back(tf);
                }
            } 
            // 2. On skipped frames, just update the boxes using fast YuNet coordinates
            else {
                for (int i = 0; i < faces.rows && i < tracked_faces.size(); i++) {
                    tracked_faces[i].box.x = static_cast<int>(faces.at<float>(i, 0));
                    tracked_faces[i].box.y = static_cast<int>(faces.at<float>(i, 1));
                    tracked_faces[i].box.width = static_cast<int>(faces.at<float>(i, 2));
                    tracked_faces[i].box.height = static_cast<int>(faces.at<float>(i, 3));
                }
            }

            // 3. Draw the results smoothly on every frame
            for (const auto& face : tracked_faces) {
                cv::rectangle(frame, face.box, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, face.label, cv::Point(face.box.x, face.box.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }

            cv::imshow("Face Detection & Encoding", frame);
            if (cv::waitKey(1) == 'q') break;
        }
    } catch (std::exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}