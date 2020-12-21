#pragma once
#include <Interpreter.hpp>
#include <MNNDefine.h>
#include <Tensor.hpp>
#include <ImageProcess.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>

using namespace std;

class evalImage {
public:
    evalImage(const char* mnn_path, int num_thread, int in_h, int in_w);
    ~evalImage();
    int inference(uint8_t* img, int originalWidth, int originalHeight);
private:
    shared_ptr<MNN::Interpreter> eval;
    MNN::Session* sess_eval = nullptr;
    MNN::Tensor* input_tensor = nullptr;
    MNN::Tensor* tensor_outputs = nullptr;
    MNN::CV::ImageProcess::Config img_config;
    shared_ptr<MNN::CV::ImageProcess> pretreat;

private:
    int resize_h;
    int resize_w;
    const char* output = "output0";
    const float mean_vals[3] = { 123.675f, 116.28f, 103.53f };
    const float norm_vals[3] = { 0.01712475383, 0.0175070028, 0.0174291939 };
};
