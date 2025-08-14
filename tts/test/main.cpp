// #include <stdio.h>
// #include "SynthesizerTrn.h"
// #include "utils.h"
// #include <string.h>
// #include <alsa/asoundlib.h> // ALSA头文件
// #include <string>
// #include "Hanz2Piny.h"
// #include "hanzi2phoneid.h"
// #include <iostream>
// #include <fstream>
// #include "ZmqServer.h"

// using namespace std;
// #include <queue>
// #include <mutex>
// #include <condition_variable>
// #include <thread>
// #include <atomic>
// #include <memory>

// #include <pthread.h>
// #include <sched.h>
// #include <sys/resource.h>

// // void set_realtime_priority()
// // {
// //     // 1. 设置线程调度策略为FIFO实时调度
// //     struct sched_param param;
// //     param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1; // 保留最高优先级给系统

// //     if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0)
// //     {
// //         perror("Warning: Failed to set real-time priority (需要root权限)");

// //         // 2. 退而求其次设置高nice值
// //         setpriority(PRIO_PROCESS, 0, -19); // 最高nice值
// //         return;
// //     }

// //     // 3. 锁定内存避免交换（可选）
// //     // mlockall(MCL_CURRENT | MCL_FUTURE);

// //     // 4. 禁用CPU频率调节（需要root）
// //     system("echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor");
// // }
// bool set_realtime_priority(pthread_t thread_id, int priority_level)
// {
//     // 参数验证
//     if (priority_level < 1 || priority_level > 99)
//     {
//         return false;
//     }

//     struct sched_param param;
//     param.sched_priority = priority_level;

//     // 先尝试设置实时调度策略
//     if (pthread_setschedparam(thread_id, SCHED_FIFO, &param) == 0)
//     {
//         return true;
//     }

//     // 失败后尝试设置RR策略
//     if (pthread_setschedparam(thread_id, SCHED_RR, &param) == 0)
//     {
//         return true;
//     }

//     // 最后尝试设置高nice值（不需要root）
//     setpriority(PRIO_PROCESS, 0, -20);
//     return (errno == 0);
// }

// zmq_component::ZmqServer server("tcp://*:7777");
// zmq_component::ZmqServer status_server("tcp://*:6677");
// bool first_msg = true;

// char *model_path_;
// float *dataW_ = nullptr;
// int32_t modelSize_ = 0;
// std::unique_ptr<SynthesizerTrn> synthesizer_;
// // 音频数据消息结构
// struct AudioMessage
// {
//     std::unique_ptr<int16_t[]> data;
//     size_t length;
//     bool is_last = false;
// };

// // 双消息队列类（文本队列和音频队列）
// class DoubleMessageQueue
// {
// public:
//     // 文本队列操作
//     void push_text(const std::string &msg)
//     {
//         {
//             std::lock_guard<std::mutex> lock(text_mutex_);
//             text_queue_.push(msg);
//         }
//         text_cond_.notify_one();
//     }

//     std::string pop_text()
//     {
//         std::unique_lock<std::mutex> lock(text_mutex_);
//         text_cond_.wait(lock, [this]
//                         { return !text_queue_.empty() || stop_; });

//         if (stop_)
//             return "";

//         std::string msg = std::move(text_queue_.front());
//         text_queue_.pop();
//         return msg;
//     }

//     // 音频队列操作
//     void push_audio(std::unique_ptr<int16_t[]> data, size_t length, bool is_last = false)
//     {
//         AudioMessage msg{std::move(data), length, is_last};
//         {
//             std::lock_guard<std::mutex> lock(audio_mutex_);
//             audio_queue_.push(std::move(msg));
//         }
//         audio_cond_.notify_one();
//     }

//     AudioMessage pop_audio()
//     {
//         std::unique_lock<std::mutex> lock(audio_mutex_);
//         audio_cond_.wait(lock, [this]
//                          { return !audio_queue_.empty() || stop_; });

//         if (stop_)
//             return {nullptr, 0, true};

//         AudioMessage msg = std::move(audio_queue_.front());
//         audio_queue_.pop();
//         return msg;
//     }

//     void stop()
//     {
//         {
//             std::lock_guard<std::mutex> lock1(text_mutex_);
//             std::lock_guard<std::mutex> lock2(audio_mutex_);
//             stop_ = true;
//         }
//         text_cond_.notify_all();
//         audio_cond_.notify_all();
//     }

// private:
//     // 文本队列
//     std::queue<std::string> text_queue_;
//     std::mutex text_mutex_;
//     std::condition_variable text_cond_;

//     std::queue<AudioMessage> audio_queue_;
//     std::mutex audio_mutex_;
//     std::condition_variable audio_cond_;

//     std::atomic<bool> stop_{false};
// };

// // ALSA音频播放函数
// void playAudioWithALSA(const int16_t *audioData, int audio_len, float speed = 1.0f)
// {
//     snd_pcm_t *pcm_handle = nullptr;
//     int err;

//     // 1. 打开PCM设备（增加错误详情）
//     if ((err = snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, 0)) < 0)
//     {
//         std::cerr << "ALSA Error: Cannot open PCM device: " << snd_strerror(err) << std::endl;
//         return;
//     }

//     // 2. 配置参数（根据语速调整采样率）
//     unsigned int sample_rate = static_cast<unsigned int>(16000 * speed);
//     if ((err = snd_pcm_set_params(pcm_handle,
//                                   SND_PCM_FORMAT_S16_LE, // 16-bit little-endian
//                                   SND_PCM_ACCESS_RW_INTERLEAVED,
//                                   1,           // Mono
//                                   sample_rate, // Adjusted sample rate
//                                   1,           // Allow software resampling
//                                   50000)) < 0)
//     { // Latency (us)
//         std::cerr << "ALSA Error: Cannot set parameters: " << snd_strerror(err) << std::endl;
//         snd_pcm_close(pcm_handle);
//         return;
//     }

//     // 3. 音频数据处理（支持慢放）
//     const snd_pcm_uframes_t frames = audio_len / 2; // 16-bit samples
//     const int max_retries = 3;
//     int retry_count = 0;

//     while (true)
//     {
//         err = snd_pcm_writei(pcm_handle, audioData, frames);
//         if (err == -EPIPE)
//         {
//             // 缓冲区欠载处理
//             std::cerr << "ALSA Warning: Buffer underrun" << std::endl;
//             if (++retry_count >= max_retries)
//                 break;
//             snd_pcm_prepare(pcm_handle);
//         }
//         else if (err < 0)
//         {
//             // 其他错误
//             std::cerr << "ALSA Error: Write failed: " << snd_strerror(err) << std::endl;
//             snd_pcm_recover(pcm_handle, err, 1);
//             break;
//         }
//         else
//         {
//             // 写入成功
//             break;
//         }
//     }

//     // 4. 清理资源（增加状态检查）
//     if (snd_pcm_state(pcm_handle) == SND_PCM_STATE_RUNNING)
//     {
//         snd_pcm_drain(pcm_handle); // 等待播放完成
//     }
//     snd_pcm_close(pcm_handle);
// }

// // 安全检查 UTF-8 字符完整性（防止截断）
// bool is_valid_utf8_continuation(uint8_t c)
// {
//     return (c & 0xC0) == 0x80; // 判断是否为 UTF-8 后续字节
// }

// std::string extract_after_think(const std::string &input)
// {
//     const std::string start_tag = "<think>";
//     const std::string end_tag = "</think>";

//     size_t start_pos = input.find(start_tag);
//     size_t end_pos = input.find(end_tag);

//     std::string result;

//     // 提取 <think>...</think> 之间的内容
//     if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos)
//     {
//         result = input.substr(start_pos + start_tag.length(), end_pos - start_pos - start_tag.length());
//     }
//     // 去掉 </think>
//     else if (end_pos != std::string::npos)
//     {
//         result = input.substr(end_pos + end_tag.length());
//     }
//     // 无标签则直接使用原字符串
//     else
//     {
//         result = input;
//     }

//     // 定义要过滤的字符（仅标点，不包含字母数字和汉字）
//     const std::string punct = " \t\n\r*#@$%^&，。：、；！？【】（）“”‘’";

//     // 安全过滤：保护 UTF-8 多字节字符
//     std::string filtered;
//     for (size_t i = 0; i < result.size();)
//     {
//         uint8_t c = result[i];
//         if (punct.find(c) != std::string::npos)
//         {
//             // 跳过标点符号
//             i++;
//         }
//         else
//         {
//             // 完整复制 UTF-8 字符（1-4字节）
//             size_t char_len = 1;
//             if ((c & 0xE0) == 0xC0)
//                 char_len = 2; // 2字节字符
//             else if ((c & 0xF0) == 0xE0)
//                 char_len = 3; // 3字节（如中文）
//             else if ((c & 0xF8) == 0xF0)
//                 char_len = 4;

//             // 检查字符完整性
//             bool is_valid = true;
//             for (size_t j = 1; j < char_len; ++j)
//             {
//                 if (i + j >= result.size() || !is_valid_utf8_continuation(result[i + j]))
//                 {
//                     is_valid = false;
//                     break;
//                 }
//             }

//             if (is_valid)
//             {
//                 filtered.append(result.substr(i, char_len));
//                 i += char_len;
//             }
//             else
//             {
//                 i++; // 跳过无效字节
//             }
//         }
//     }

//     return filtered;
// }
// // TTS合成线程函数
// void synthesis_worker(DoubleMessageQueue &queue, const std::string &model_path)
// {
//     // set_realtime_priority(pthread_self(), 99);

//     // float *dataW = nullptr;
//     // 方案1：使用vector创建可修改副本
//     // std::vector<char> model_path_copy(model_path.begin(), model_path.end());
//     // model_path_copy.push_back('\0');

//     // int32_t modelSize = ttsLoadModel(model_path_copy.data(), &dataW);
//     // // int32_t modelSize = ttsLoadModel(model_path.c_str(), &dataW);
//     // SynthesizerTrn synthesizer(dataW, modelSize);

//     while (true)
//     {
//         std::string text = queue.pop_text();
//         if (text.empty())
//             break; // 停止信号
//         if (text.find("END") != std::string::npos)
//         {
//             first_msg = true;
//             // status_server.send("Echo: play end success");

//             std::cout << "Detected END signal in received message" << std::endl;
//             // 提取END之前的内容
//             size_t end_pos = text.find("END");
//             text = text.substr(0, end_pos);
//         }

//         int32_t audio_len = 0;
//         if (!text.empty())
//         {
//             std::cout << "infer ing !!" << text << std::endl;

//             int16_t *wavData = synthesizer_->infer(text, 0, 1.0, audio_len);
//             std::cout << "infer success !!" << std::endl;

//             // 将音频数据放入音频队列
//             auto audio_data = std::make_unique<int16_t[]>(audio_len);
//             memcpy(audio_data.get(), wavData, audio_len * sizeof(int16_t));
//             queue.push_audio(std::move(audio_data), audio_len, first_msg);

//             tts_free_data(wavData);
//         }
//         else
//         {
//             // 创建空音频数据
//             audio_len = 0;                                     // 确保 audio_len 被正确设置为 0
//             auto empty_audio = std::make_unique<int16_t[]>(0); // 长度为 0 的数组
//             queue.push_audio(std::move(empty_audio), 0, first_msg);
//         }
//     }

//     // tts_free_data(dataW);
// }

// // 音频播放线程函数
// void playback_worker(DoubleMessageQueue &queue)
// {
//     while (true)
//     {
//         auto msg = queue.pop_audio();
//         if (msg.data == nullptr)
//             break; // 停止信号
//         playAudioWithALSA(msg.data.get(), msg.length * sizeof(int16_t), 1.0f);
//         if (msg.is_last)
//         {
//             status_server.send("Echo: play end success");
//         }
//     }
// }

// std::vector<std::string> split_long_text(const std::string &text, size_t max_length = 10)
// {
//     std::vector<std::string> segments;
//     int size = text.length();
//     // 如果文本本身不长，直接返回
//     if (size <= max_length)
//     {
//         segments.push_back(text);
//         return segments;
//     }

//     segments.push_back(text.substr(0, 10));
//     segments.push_back(text.substr(10, size));
    
//     return segments;

// }

// int main(int argc, char **argv)
//     {
//         if (argc < 3)
//         {
//             std::cerr << "Usage: " << argv[0] << " <zmq_port> <model_path>" << std::endl;
//             return 1;
//         }
//         model_path_ = argv[2];
//         modelSize_ = ttsLoadModel(model_path_, &dataW_);
//         synthesizer_ = std::make_unique<SynthesizerTrn>(dataW_, modelSize_);
//         DoubleMessageQueue queue;

//         // 启动工作线程
//         std::thread synthesis_thread(synthesis_worker, std::ref(queue), std::string(argv[2]));
//         std::thread playback_thread(playback_worker, std::ref(queue));

//         while (true)
//         {
//             try
//             {

//                 if (first_msg)
//                 {
//                     std::string req = status_server.receive();
//                     std::cout << "111111status sever received: " << req << std::endl;
//                 }
//                 first_msg = false;

//                 std::string text = server.receive();

//                 server.send("Echo: received");
//                 std::cout << "oriange Server received: " << text << std::endl;

//                 // text = extract_after_think(text);
//                 if (!text.empty() && text.find("<think>") == std::string::npos)
//                 {
//                     std::cout << "Text received: " << text << std::endl;

//                     // 将文本放入队列
//                     queue.push_text(text);
//                 }
//             }
//             catch (const std::exception &e)
//             {
//                 std::cerr << "Error: " << e.what() << std::endl;
//             }
//         }

//         // 清理
//         queue.stop();
//         synthesis_thread.join();
//         playback_thread.join();
//         return 0;
//     }