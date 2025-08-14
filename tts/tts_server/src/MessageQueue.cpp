#include "MessageQueue.h"

void DoubleMessageQueue::push_text(const std::string &msg)
{
    {
        std::lock_guard<std::mutex> lock(text_mutex_);
        text_queue_.push(msg);
    }
    text_cond_.notify_one();
}

std::string DoubleMessageQueue::pop_text()
{
    std::unique_lock<std::mutex> lock(text_mutex_);
    text_cond_.wait(lock, [this]
                    { return !text_queue_.empty() || stop_; });

    if (stop_)
        return "";

    std::string msg = std::move(text_queue_.front());
    text_queue_.pop();
    return msg;
}

void DoubleMessageQueue::push_audio(std::unique_ptr<int16_t[]> data, size_t length, bool is_last)
{
    AudioMessage msg{std::move(data), length, is_last};
    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        audio_queue_.push(std::move(msg));
    }
    audio_cond_.notify_one();
}

AudioMessage DoubleMessageQueue::pop_audio()
{
    std::unique_lock<std::mutex> lock(audio_mutex_);
    audio_cond_.wait(lock, [this]
                     { return !audio_queue_.empty() || stop_; });

    if (stop_)
        return {nullptr, 0, true};

    AudioMessage msg = std::move(audio_queue_.front());
    audio_queue_.pop();
    return msg;
}

void DoubleMessageQueue::stop()
{
    {
        std::lock_guard<std::mutex> lock1(text_mutex_);
        std::lock_guard<std::mutex> lock2(audio_mutex_);
        stop_ = true;
    }
    text_cond_.notify_all();
    audio_cond_.notify_all();
}