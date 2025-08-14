#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <string>
#include <vector>

class TextProcessor {
public:
    static std::string extract_after_think(const std::string &input);
    static std::string clean_text(const std::string &text);
    static std::vector<std::string> process_input_text(const std::string &input);
};

#endif // TEXT_PROCESSOR_H