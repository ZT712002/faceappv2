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

// Distance calculation helper function
double calculate_distance(const dlib::matrix<float, 0, 1>& v1, const dlib::matrix<float, 0, 1>& v2) {
    return dlib::length(v1 - v2);
}

int main() {
    try {
        string yunetPath = "C:/Users/Zzz/Documents/fad/models/face_detection_yunet_2023mar.onnx";
        cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(yunetPath, "", cv::Size(320, 320));
        if (detector.empty()) throw std::runtime_error("Failed to load YuNet model.");

        shape_predictor sp;
        deserialize("C:/Users/Zzz/Documents/fad/models/shape_predictor_5_face_landmarks.dat") >> sp;
        
        anet_type net;
        deserialize("C:/Users/Zzz/Documents/fad/models/dlib_face_recognition_resnet_model_v1.dat") >> net;

        string myFacePath = R"(C:\Users\Zzz\Documents\fad\faces\test.jpg)"; 
        cv::Mat myOriginalFrame = cv::imread(myFacePath);
        
        if (myOriginalFrame.empty()) {
            cerr << "Error: Could not load desktop image from " << myFacePath << endl;
            cerr << "Please check if the file extension is exactly .jpg and not .jpeg or .png" << endl;
            return -1;
        }

        // Resize the portrait image if it's too massive. YuNet operates best at standard dimensions.
        cv::Mat myFrame;
        int targetWidth = 640;
        int targetHeight = static_cast<int>(myOriginalFrame.rows * (static_cast<float>(targetWidth) / myOriginalFrame.cols));
        cv::resize(myOriginalFrame, myFrame, cv::Size(targetWidth, targetHeight));

        cv::Mat myFaces;
        detector->setInputSize(myFrame.size());
        detector->detect(myFrame, myFaces);

        if (myFaces.rows == 0) {
            cerr << "Error: Still no face detected. Attempting lower confidence threshold fallback..." << endl;
            
            // Temporary threshold adjustment fallback just for this image initialization step
            auto fallbackDetector = cv::FaceDetectorYN::create(yunetPath, "", myFrame.size(), 0.5f, 0.3f);
            fallbackDetector->detect(myFrame, myFaces);
            
            if (myFaces.rows == 0) {
                cerr << "Critical Error: YuNet cannot find a face. Ensure the photo isn't heavily cropped or sideways." << endl;
                return -1;
            }
        }

        // Generate 128D descriptor vector for the reference face
        dlib::cv_image<bgr_pixel> dlib_my_img(myFrame);
        int mx = static_cast<int>(myFaces.at<float>(0, 0));
        int my = static_cast<int>(myFaces.at<float>(0, 1));
        int mw = static_cast<int>(myFaces.at<float>(0, 2));
        int mh = static_cast<int>(myFaces.at<float>(0, 3));
        
        dlib::rectangle d_my_rect(mx, my, mx + mw, my + mh);
        auto my_shape = sp(dlib_my_img, d_my_rect);
        
        dlib::matrix<rgb_pixel> my_face_chip;
        extract_image_chip(dlib_my_img, get_face_chip_details(my_shape, 150, 0.25), my_face_chip);
        
        dlib::matrix<float, 0, 1> my_reference_descriptor = net(my_face_chip);
        cout << "Successfully encoded your desktop reference face!" << endl;

        // 4. Set Up Webcam Pipeline
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        detector->setInputSize(cv::Size(640, 480));

        cv::Mat frame;
        int frame_count = 0;
        
        struct TrackedFace {
            cv::Rect box;
            string label = "Processing...";
            cv::Scalar color = cv::Scalar(0, 0, 255); // Default color (red)
        };
        std::vector<TrackedFace> tracked_faces;

        while (cap.read(frame)) {
            frame_count++;
            cv::Mat faces;
            detector->detect(frame, faces);

            // Trigger complete processing context (landmarks + face verification execution) on target frames
            if (frame_count % 6 == 0 || tracked_faces.size() != faces.rows) { 
                tracked_faces.clear();
                cv_image<bgr_pixel> dlib_img(frame);

                for (int i = 0; i < faces.rows; i++) {
                    int x = static_cast<int>(faces.at<float>(i, 0));
                    int y = static_cast<int>(faces.at<float>(i, 1));
                    int w = static_cast<int>(faces.at<float>(i, 2));
                    int h = static_cast<int>(faces.at<float>(i, 3));

                    dlib::rectangle d_rect(x, y, x + w, y + h);
                    if (d_rect.is_empty()) continue;

                    auto shape = sp(dlib_img, d_rect);
                    matrix<rgb_pixel> face_chip;
                    extract_image_chip(dlib_img, get_face_chip_details(shape, 150, 0.25), face_chip);
                    
                    // Heavy mathematical feature space representation layer extraction
                    matrix<float, 0, 1> face_descriptor = net(face_chip); 

                    // Check similarity metric threshold 
                    double distance = calculate_distance(face_descriptor, my_reference_descriptor);
                    
                    TrackedFace tf;
                    tf.box = cv::Rect(x, y, w, h);
                    
                    // Match boundary threshold context evaluation (Dlib target recommendation threshold: 0.6)
                    if (distance < 0.4) {
                        tf.label = "Zzz Match (" + to_string(distance).substr(0, 4) + ")";
                        tf.color = cv::Scalar(0, 255, 0); // Green box
                    } else {
                        tf.label = "Unknown (" + to_string(distance).substr(0, 4) + ")";
                        tf.color = cv::Scalar(0, 0, 255); // Red box
                    }
                    tracked_faces.push_back(tf);
                }
            } 
            // On skipped frames, smoothly pass geometric detection coordinates directly into display layer tracking structures
            else {
                for (int i = 0; i < faces.rows && i < tracked_faces.size(); i++) {
                    tracked_faces[i].box.x = static_cast<int>(faces.at<float>(i, 0));
                    tracked_faces[i].box.y = static_cast<int>(faces.at<float>(i, 1));
                    tracked_faces[i].box.width = static_cast<int>(faces.at<float>(i, 2));
                    tracked_faces[i].box.height = static_cast<int>(faces.at<float>(i, 3));
                }
            }

            // Draw results on screen dynamically
            for (const auto& face : tracked_faces) {
                cv::rectangle(frame, face.box, face.color, 2);
                cv::putText(frame, face.label, cv::Point(face.box.x, face.box.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, face.color, 1);
            }

            cv::imshow("Face Detection & Encoding", frame);
            if (cv::waitKey(1) == 'q') break;
        }
    } catch (std::exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}