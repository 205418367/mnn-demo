#include "UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    string image_file = "../imgs/1.jpg";
    string mnn_path = "../models/RFB-320-quant-KL-5792.mnn";

    UltraFace ultraface(mnn_path, 320, 240, 4, 0.65); 
    cv::Mat frame = cv::imread(image_file);
    auto start = chrono::steady_clock::now();
    vector<FaceInfo> face_info;

    int image_w = frame.cols;
    int image_h = frame.rows;
    ultraface.detect(frame, face_info);
    for (auto face : face_info) {
        cv::Point pt1(face.x1, face.y1);
        cv::Point pt2(face.x2, face.y2);
        cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "infer time: " << elapsed.count() << " s" << endl;
    cv::imshow("UltraFace", frame);
    cv::waitKey();
    return 0;
}
