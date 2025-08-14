#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <pthread.h>
#include <vector>

namespace utils
{
    bool set_realtime_priority(pthread_t thread_id, int priority_level);
    bool is_valid_utf8_continuation(uint8_t c);
    std::vector<std::string> split_long_text(const std::string &text, size_t max_length = 10);
}

#endif // UTILS_H