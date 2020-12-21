#include "imagenet.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


int main() {
    const char* model_path = "../models/imagenet/efficientnet_b0_quan.mnn";
    const std::string image_path = "../imgs/imagenet/donut.jpg";
    const std::string labels = "../imgs/imagenet/labels.txt";
    
    vector<string> txt;
    ifstream infile(labels);
    string temp;
    while(getline(infile, temp)){
      	txt.push_back(temp);
    }

    evalImage evalimage(model_path, 4, 224, 224); 
    cv::Mat frame = cv::imread(image_path);
    int idx = evalimage.inference(frame);
    
    std::cout<<"##succ## "<<txt[idx]<<std::endl;
    return 0;
}   
