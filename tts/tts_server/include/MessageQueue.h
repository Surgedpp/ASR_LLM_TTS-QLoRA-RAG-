#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <cstdint>

struct AudioMessage {
    std::unique_ptr<int16_t[]> data;
    size_t length;
    bool is_last = false;
};

class DoubleMessageQueue {
public:
    void push_text(const std::string &msg);
    std::string pop_text();
    
    void push_audio(std::unique_ptr<int16_t[]> data, size_t length, bool is_last = false);
    AudioMessage pop_audio();
    
    void stop();

private:
    std::queue<std::string> text_queue_;
    std::mutex text_mutex_;
    std::condition_variable text_cond_;

    std::queue<AudioMessage> audio_queue_;
    std::mutex audio_mutex_;
    std::condition_variable audio_cond_;

    std::atomic<bool> stop_{false};
};

#endif // MESSAGE_QUEUE_H