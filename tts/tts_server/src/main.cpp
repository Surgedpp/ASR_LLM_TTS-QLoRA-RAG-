#include "TTSModel.h"
#include "MessageQueue.h"
#include "AudioPlayer.h"
#include "TextProcessor.h"
#include "Utils.h"
#include "ZmqServer.h"

#include <thread>
#include <iostream>
#include <atomic>
#include <memory>

zmq_component::ZmqServer server("tcp://*:7777");
zmq_component::ZmqServer status_server("tcp://*:6677");
std::atomic<bool> first_msg(true);

void synthesis_worker(DoubleMessageQueue &queue, TTSModel &model) {
    // utils::set_realtime_priority(pthread_self(), 99);

    while (true) {
        std::string text = queue.pop_text();
        if (text.empty()) break;

        if (text.find("END") != std::string::npos) {
            first_msg = true;
            size_t end_pos = text.find("END");
            text = text.substr(0, end_pos);
        }

        int32_t audio_len = 0;
        if (!text.empty()) {
            std::cout << "[TTS infer] Inferring text: " << text << std::endl;
            int16_t* wavData = model.infer(text, audio_len);
            
            if (wavData && audio_len > 0) {
                auto audio_data = std::make_unique<int16_t[]>(audio_len);
                memcpy(audio_data.get(), wavData, audio_len * sizeof(int16_t));
                queue.push_audio(std::move(audio_data), audio_len, first_msg);
                model.free_data(wavData);
            }
        } else {
            auto empty_audio = std::make_unique<int16_t[]>(0);
            queue.push_audio(std::move(empty_audio), 0, first_msg);
        }
    }
}

void playback_worker(DoubleMessageQueue &queue, AudioPlayer &player) {
    while (true) {
        auto msg = queue.pop_audio();
        if (msg.data == nullptr) break;
        
        player.play(msg.data.get(), msg.length * sizeof(int16_t), 1.0f);
        
        if (msg.is_last) {
            status_server.send("[tts -> voice]play end success");
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    try {
    
        TTSModel model(argv[1]);
        AudioPlayer player;
        DoubleMessageQueue queue;

        std::thread synthesis_thread(synthesis_worker, std::ref(queue), std::ref(model));
        std::thread playback_thread(playback_worker, std::ref(queue), std::ref(player));

        while (true) {
            if (first_msg) {
                std::string req = status_server.receive();
                std::cout << "[voice -> tts] received: " << req << std::endl;
            }
            first_msg = false;

            std::string text = server.receive();
            server.send("Echo: received");
            std::cout << "[llm -> tts] received: " << text << std::endl;

            if (!text.empty() && text.find("<think>") == std::string::npos) {
              
    
                queue.push_text(text);
            }
        }

        // 清理
        queue.stop();
        synthesis_thread.join();
        playback_thread.join();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}