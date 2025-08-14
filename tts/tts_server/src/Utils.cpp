#include "Utils.h"
#include <sys/resource.h>
#include <algorithm>

namespace utils {

bool set_realtime_priority(pthread_t thread_id, int priority_level) {
    if (priority_level < 1 || priority_level > 99) {
        return false;
    }

    struct sched_param param;
    param.sched_priority = priority_level;

    if (pthread_setschedparam(thread_id, SCHED_FIFO, &param) == 0) {
        return true;
    }

    if (pthread_setschedparam(thread_id, SCHED_RR, &param) == 0) {
        return true;
    }

    setpriority(PRIO_PROCESS, 0, -20);
    return (errno == 0);
}

bool is_valid_utf8_continuation(uint8_t c) {
    return (c & 0xC0) == 0x80;
}

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