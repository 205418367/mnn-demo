#include "imagenet.hpp"
#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


int main() {
    const char* model_path = "../models/imagenet/efficientnet_b0_quan.mnn";
    const char* image_path = "../imgs/imagenet/cls_001.jpg";
    const std::string labels = "../imgs/imagenet/labels.txt";
    
    vector<string> txt;
    ifstream infile(labels);
    string temp;
    while(getline(infile, temp)){
      	txt.push_back(temp);
    }

    evalImage evalimage(model_path, 4, 224, 224); 
    int originalWidth;
    int originalHeight;
    int originChannel;
    auto frame = stbi_load(image_path, &originalWidth, &originalHeight, &originChannel,3);
    int idx = evalimage.inference((uint8_t*)frame, originalWidth, originalHeight);
    
    //stbi_write_png(outputImageFileName, originalWidth, originalHeight, 4, inputImage, 4 * originalWidth);
    //stbi_image_free(inputImage);
    std::cout<<"##succ## "<<txt[idx]<<std::endl;
    return 0;
}   
