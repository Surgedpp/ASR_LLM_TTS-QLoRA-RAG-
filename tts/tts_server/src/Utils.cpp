#include "Utils.h"
#include <sys/resource.h>
#include <algorithm>

namespace utils {
// 设置线程优先级，如为音频播放线程设置为高优先级，降低音频延迟，避免卡顿，提高响应速度
bool set_realtime_priority(pthread_t thread_id, int priority_level) {
    // 优先级范围1-99，普通进程优先级100-139
    if (priority_level < 1 || priority_level > 99) {
        return false;
    }

    struct sched_param param;
    param.sched_priority = priority_level;
    //同优先级时：SCHED_FIFO > SCHED_RR     不同优先级时：谁的 rt_priority 数值大，谁优先级高
    if (pthread_setschedparam(thread_id, SCHED_FIFO, &param) == 0) {    //实时先进先出，最高优先级
        return true;
    }

    if (pthread_setschedparam(thread_id, SCHED_RR, &param) == 0) {  //实时轮转，高优先级
        return true;
    }
    //int setpriority(int which, id_t who, int prio);
//                ^^^^^^^^^^^^  ^^^^^^^^  ^^^^^
//                目标类型      目标ID     优先级
//  普通进程采取CFS策略，优先级由nice值决定（-20~19），nice值越低，优先级越高
    setpriority(PRIO_PROCESS, 0, -20);
    return (errno == 0);
}

bool is_valid_utf8_continuation(uint8_t c) {
    return (c & 0xC0) == 0x80;
}

//文本分割函数，将长文本切为指定长度的短文本，防止TTS模型处理时内存限制、推理超时，实现流式输出
std::vector<std::string> split_long_text(const std::string &text, size_t max_length) {
    std::vector<std::string> segments;
    size_t size = text.length();
    
    if (size <= max_length) {
        segments.push_back(text);
        return segments;
    }

    for (size_t i = 0; i < size; i += max_length) {
        segments.push_back(text.substr(i, max_length));
    }
    
    return segments;
}

} // namespace utils