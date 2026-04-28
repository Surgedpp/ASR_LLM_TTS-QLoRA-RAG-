#include "TTSModel.h"
#include "SynthesizerTrn.h"
#include "utils.h"


TTSModel::TTSModel(const std::string &model_path)
{
    load_model(model_path);
}

TTSModel::~TTSModel()
{
    if (dataW_)
    {
        tts_free_data(reinterpret_cast<int16_t *>(dataW_));
    }
}

bool TTSModel::load_model(const std::string &model_path)
{
    std::vector<char> model_path_copy(model_path.begin(), model_path.end());
    model_path_copy.push_back('\0');

    modelSize_ = ttsLoadModel(model_path_copy.data(), &dataW_);
    if (modelSize_ <= 0 || !dataW_)
    {
        return false;
    }
    synthesizer_ = std::make_unique<SynthesizerTrn>(dataW_, modelSize_); //智能指针unique对象
    return true;
}

int16_t *TTSModel::infer(const std::string &text, int32_t &audio_len)
{
    if (!synthesizer_)
        return nullptr;
    //infer(输入文本，说话人ID，语速缩放因子，输出的音频长度)
    return synthesizer_->infer(text, 0, 1.0, audio_len);
}

void TTSModel::free_data(int16_t *data)
{
    tts_free_data(data);
}