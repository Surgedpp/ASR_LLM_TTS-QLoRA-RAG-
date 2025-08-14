#ifndef TTS_MODEL_H
#define TTS_MODEL_H

#include <memory>
#include <string>
#include <stdio.h>
#include "SynthesizerTrn.h"
#include "utils.h"
#include "Hanz2Piny.h"
#include "hanzi2phoneid.h"
#include <iostream>
#include <fstream>

// 前向声明
class SynthesizerTrn;

class TTSModel {
public:
    explicit TTSModel(const std::string& model_path);
    ~TTSModel();
    
    bool load_model(const std::string& model_path);
    int16_t* infer(const std::string& text, int32_t& audio_len);
    void free_data(int16_t* data);
    
private:
    float* dataW_ = nullptr;
    int32_t modelSize_ = 0;
    std::unique_ptr<SynthesizerTrn> synthesizer_;
};

#endif // TTS_MODEL_H