#include "TextProcessor.h"
#include "Utils.h"
#include <algorithm>

std::string TextProcessor::extract_after_think(const std::string &input)
{
    const std::string start_tag = "<think>";
    const std::string end_tag = "</think>";

    size_t start_pos = input.find(start_tag);
    size_t end_pos = input.find(end_tag);

    std::string result;

    if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos)
    {
        result = input.substr(start_pos + start_tag.length(), end_pos - start_pos - start_tag.length());
    }
    else if (end_pos != std::string::npos)
    {
        result = input.substr(end_pos + end_tag.length());
    }
    else
    {
        result = input;
    }

    return result;
}

std::string TextProcessor::clean_text(const std::string &text)
{
    const std::string punct = " \t\n\r*#@$%^&，。：、；！？【】（）“”‘’";
    std::string filtered;

    for (size_t i = 0; i < text.size();)
    {
        uint8_t c = text[i];
        if (punct.find(c) != std::string::npos)
        {
            i++;
        }
        else
        {
            size_t char_len = 1;
            if ((c & 0xE0) == 0xC0)
                char_len = 2;
            else if ((c & 0xF0) == 0xE0)
                char_len = 3;
            else if ((c & 0xF8) == 0xF0)
                char_len = 4;

            bool is_valid = true;
            for (size_t j = 1; j < char_len; ++j)
            {
                if (i + j >= text.size() || !utils::is_valid_utf8_continuation(text[i + j]))
                {
                    is_valid = false;
                    break;
                }
            }

            if (is_valid)
            {
                filtered.append(text.substr(i, char_len));
                i += char_len;
            }
            else
            {
                i++;
            }
        }
    }

    return filtered;
}

std::vector<std::string> TextProcessor::process_input_text(const std::string &input)
{
    // std::string processed = extract_after_think(input);
    std::string processed = input;
    processed = clean_text(processed);
    return utils::split_long_text(processed);
}