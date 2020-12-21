#include "imagenet.hpp"

evalImage::evalImage(const char* modelfile, int num_thread, int in_h, int in_w):resize_h(in_h),resize_w(in_w)
{
    eval = shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelfile));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;    

    sess_eval = eval->createSession(config);
    input_tensor = eval->getSessionInput(sess_eval, NULL);
    tensor_outputs = eval->getSessionOutput(sess_eval, output);
    
    ::memcpy(img_config.mean, mean_vals, sizeof(mean_vals));
    ::memcpy(img_config.normal, norm_vals, sizeof(norm_vals));
    img_config.sourceFormat = MNN::CV::ImageFormat::BGR;
    img_config.destFormat = MNN::CV::ImageFormat::RGB;
    img_config.filterType = MNN::CV::Filter::BICUBIC;
    img_config.wrap = (MNN::CV::Wrap)(2);
    pretreat = shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
}  


evalImage::~evalImage()
{
   eval->releaseModel();
   eval->releaseSession(sess_eval);
}
    
int evalImage::inference(cv::Mat& img)
{
   Mat image;
   resize(img, image, Size(resize_h, resize_w));
   //eval->resizeTensor(input_tensor, {1, 3, resize_h, resize_w});
   //eval->resizeSession(sess_eval);
   
   pretreat->convert(image.data, resize_w, resize_h, image.step[0], input_tensor);
   eval->runSession(sess_eval);
   
   MNN::Tensor tensor_outputs_host(tensor_outputs, tensor_outputs->getDimensionType());
   tensor_outputs->copyToHostTensor(&tensor_outputs_host);
   float* out = tensor_outputs->host<float>();

   int max_index=0;
   float max_value=0.0;
   for (int i=0; i<1001; i++){
       if (*(out+i) > max_value){
           max_value = *(out+i);
           max_index = i;
       }     
   }    
   return max_index;
}
