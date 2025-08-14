#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace edge_llm_rag
{

    struct QueryFeatures
    {
        std::vector<std::string> keywords;
        float urgency_score;               
        float complexity_score;            
        float factual_score;               
        float creative_score;              
        int query_length;                
        bool contains_question_words;      
        bool contains_emergency_words;     
        bool contains_technical_words;   
    };

    struct QueryClassification
    {
        enum QueryType
        {
            FACTUAL_QUERY,   
            COMPLEX_QUERY,   
            CREATIVE_QUERY,  
            EMERGENCY_QUERY, 
            UNKNOWN_QUERY   
        };

        QueryType query_type;
        float confidence;
        std::string reasoning;
        bool requires_immediate_response;
    };


    class QueryClassifier
    {
    public:
        explicit QueryClassifier();

        QueryFeatures analyze_query_features(const std::string &query);

        QueryClassification classify_query(const std::string &query);

        std::vector<std::string> extract_keywords(const std::string &query);
        std::string determine_domain(const std::string &query,
                                     const std::vector<std::string> &keywords);

    private:
        std::unordered_map<std::string, std::vector<std::string>> keyword_dict_;

        void initialize_keyword_dictionary();

        float calculate_urgency_score(const std::vector<std::string> &keywords);

        float calculate_complexity_score(const std::string &query,
                                         const std::vector<std::string> &keywords);

        float calculate_factual_score(const std::vector<std::string> &keywords);

        float calculate_creative_score(const std::vector<std::string> &keywords);

        bool detect_question_words(const std::string &query);

        bool detect_emergency_words(const std::string &query);

        bool detect_technical_words(const std::string &query);

    };

  
} // namespace edge_llm_rag