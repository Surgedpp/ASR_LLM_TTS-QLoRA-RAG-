#include "query_classifier.h"
#include <algorithm>
#include <regex>
#include <iostream>

namespace edge_llm_rag
{

    QueryClassifier::QueryClassifier()
    {
        initialize_keyword_dictionary();
    }

    QueryFeatures QueryClassifier::analyze_query_features(const std::string &query)
    {
        QueryFeatures features;
        features.query_length = static_cast<int>(query.length());

        features.keywords = extract_keywords(query);

        features.urgency_score = calculate_urgency_score(features.keywords);
        features.complexity_score = calculate_complexity_score(query, features.keywords);
        features.factual_score = calculate_factual_score(features.keywords);
        features.creative_score = calculate_creative_score(features.keywords);

        features.contains_question_words = detect_question_words(query);
        features.contains_emergency_words = detect_emergency_words(query);
        features.contains_technical_words = detect_technical_words(query);

        return features;
    }

    QueryClassification QueryClassifier::classify_query(const std::string &query)
    {
        QueryFeatures features = analyze_query_features(query);

        QueryClassification classification;

        classification.requires_immediate_response = features.urgency_score > 0.7f;

        if (features.urgency_score > 0.7f || features.contains_emergency_words)
        {
            classification.query_type = QueryClassification::EMERGENCY_QUERY;
        }
        else if (features.factual_score >= 0.5f)
        {
            classification.query_type = QueryClassification::FACTUAL_QUERY;
        }
        else if (features.creative_score > 0.6f)
        {
            classification.query_type = QueryClassification::CREATIVE_QUERY;
        }
        else if (features.complexity_score > 0.6f)
        {
            classification.query_type = QueryClassification::COMPLEX_QUERY;
        }
        else
        {
            classification.query_type = QueryClassification::UNKNOWN_QUERY;
        }

        return classification;
    }

    void QueryClassifier::initialize_keyword_dictionary()
    {
        keyword_dict_["emergency"] = {
            "故障", "警告", "危险", "紧急", "异常", "失灵", "失效", "损坏",
            "发动机故障", "制动故障", "转向故障", "电气故障", "安全气囊", "ABS故障"};

        keyword_dict_["technical"] = {
            "发动机", "制动", "变速箱", "电气", "空调", "转向", "悬挂", "轮胎",
            "机油", "冷却液", "制动液", "变速箱油", "电瓶", "发电机", "起动机"};

        keyword_dict_["maintenance"] = {
            "保养", "维修", "更换", "检查", "清洁", "调整", "润滑", "紧固",
            "定期保养", "机油更换", "滤清器", "火花塞", "制动片", "轮胎更换"};

        keyword_dict_["feature"] = {
            "自动泊车", "车道保持", "定速巡航", "导航", "娱乐", "空调控制",
            "座椅调节", "后视镜", "雨刷", "灯光", "音响", "蓝牙"};

        keyword_dict_["question"] = {
            "什么", "怎么", "如何", "为什么", "哪里", "何时", "多少", "哪个",
            "吗", "呢", "嘛", "能不能", "可不可以", "有没有", "推荐一下", "怎么去", "去哪里", "怎么玩"};

        keyword_dict_["creative"] = {
            "推荐", "建议", "想法", "创意", "优化", "改进", "设计", "规划",
            "旅游", "旅行", "出行", "景点", "门票", "酒店", "民宿", "机票", "火车票", "高铁",
            "行程", "路线", "攻略", "签证", "租车", "自驾", "海岛", "海滩", "公园", "博物馆",
            "古镇", "温泉", "夜市", "特产", "美食", "摄影", "网红", "打卡", "露营", "徒步",
            "游玩", "娱乐", "主题乐园", "游乐园", "迪士尼", "环球影城", "水上乐园",
            "演唱会", "音乐节", "展览", "赛事", "滑雪", "潜水", "骑行", "登山",
            "预订", "订票", "订酒店", "退改签", "行李", "登机", "值机", "改签", "延误", "转机",
            "天气", "笑话", "故事", "新闻", "百科", "科普", "翻译", "计算", "单位换算",
            "今天", "明天", "现在", "附近", "哪里有", "怎么走"};
    }

    std::vector<std::string> QueryClassifier::extract_keywords(const std::string &query)
    {
        std::vector<std::string> keywords;

        for (const auto &[category, words] : keyword_dict_)
        {
            for (const auto &word : words)
            {
                if (query.find(word) != std::string::npos)
                {
                    keywords.push_back(word);
                }
            }
        }

        return keywords;
    }

    float QueryClassifier::calculate_urgency_score(const std::vector<std::string> &keywords)
    {
        float score = 0.0f;
        int emergency_count = 0;

        for (const auto &keyword : keywords)
        {
            if (std::find(keyword_dict_["emergency"].begin(),
                          keyword_dict_["emergency"].end(), keyword) != keyword_dict_["emergency"].end())
            {
                emergency_count++;
            }
        }

        score = std::min(1.0f, static_cast<float>(emergency_count) * 0.3f);
        return score;
    }

    float QueryClassifier::calculate_complexity_score(const std::string &query,
                                                      const std::vector<std::string> &keywords)
    {
        float score = 0.0f;

        // 查询长度（30%）
        score += std::min(1.0f, static_cast<float>(query.length()) / 100.0f) * 0.3f;

        // 关键词数量（40%）
        score += std::min(1.0f, static_cast<float>(keywords.size()) / 10.0f) * 0.4f;

        // 技术词汇比例（30%）
        int technical_count = 0;
        for (const auto &keyword : keywords)
        {
            if (std::find(keyword_dict_["technical"].begin(),
                          keyword_dict_["technical"].end(), keyword) != keyword_dict_["technical"].end())
            {
                technical_count++;
            }
        }
        score += std::min(1.0f, static_cast<float>(technical_count) / 5.0f) * 0.3f;

        return std::min(1.0f, score);
    }

    float QueryClassifier::calculate_factual_score(const std::vector<std::string> &keywords)
    {
        float score = 0.0f;

        for (const auto &keyword : keywords)
        {
            if (std::find(keyword_dict_["technical"].begin(),
                          keyword_dict_["technical"].end(), keyword) != keyword_dict_["technical"].end())
            {
                score += 0.4f;
            }
            if (std::find(keyword_dict_["maintenance"].begin(),
                          keyword_dict_["maintenance"].end(), keyword) != keyword_dict_["maintenance"].end())
            {
                score += 0.4f;
            }

            if (std::find(keyword_dict_["feature"].begin(),
                          keyword_dict_["feature"].end(), keyword) != keyword_dict_["feature"].end())
            {
                score += 0.5f;
            }
        }

        return std::min(1.0f, score);
    }

    float QueryClassifier::calculate_creative_score(const std::vector<std::string> &keywords)
    {
        float score = 0.0f;

        for (const auto &keyword : keywords)
        {
            if (std::find(keyword_dict_["creative"].begin(),
                          keyword_dict_["creative"].end(), keyword) != keyword_dict_["creative"].end())
            {
                score += 0.3f;
            }
        }

        return std::min(1.0f, score);
    }

    bool QueryClassifier::detect_question_words(const std::string &query)
    {
        for (const auto &word : keyword_dict_["question"])
        {
            if (query.find(word) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    bool QueryClassifier::detect_emergency_words(const std::string &query)
    {
        for (const auto &word : keyword_dict_["emergency"])
        {
            if (query.find(word) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    bool QueryClassifier::detect_technical_words(const std::string &query)
    {
        for (const auto &word : keyword_dict_["technical"])
        {
            if (query.find(word) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

} // namespace edge_llm_rag
