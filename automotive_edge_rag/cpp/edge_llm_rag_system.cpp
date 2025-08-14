#pragma once
#include "edge_llm_rag_system.h"
#include "query_classifier.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>
#include <regex>
#include <cwchar>
#include <cstdlib>
#include <codecvt>

namespace edge_llm_rag
{
    EdgeLLMRAGSystem::EdgeLLMRAGSystem()
        :  is_initialized_(false)
    {

        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path");
        path.append("python");

        py::module_ mod = py::module_::import("vehicle_vector_search");
        searcher = mod.attr("VehicleVectorSearch")("vector_db");

        std::cout << "Loading model once..." << std::endl;
        auto load_t0 = std::chrono::high_resolution_clock::now();

        fs::path cpp_dir = fs::absolute(__FILE__).parent_path();
        fs::path model_path = cpp_dir.parent_path() / "models";
        searcher.attr("load_model")(model_path.string());
        // searcher.attr("load_model")();
        auto load_t1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
        std::cout << "Model loaded (" << std::fixed << std::setprecision(2) << load_ms << " ms)" << std::endl;

        // 打印统计信息
        py::object stats = searcher.attr("get_statistics")();
        std::cout << "Stats: total_documents=" << stats["total_documents"].cast<int>()
                  << ", embedding_dimension=" << stats["embedding_dimension"].cast<int>()
                  << std::endl;
    }

    EdgeLLMRAGSystem::~EdgeLLMRAGSystem()
    {
        
    }

    bool EdgeLLMRAGSystem::initialize()
    {
        try
        {
            query_classifier_ = std::make_unique<QueryClassifier>();

            query_cache_.clear();

            is_initialized_ = true;
            std::cout << "系统初始化成功" << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "系统初始化失败: " << e.what() << std::endl;
            return false;
        }
    }

    std::string EdgeLLMRAGSystem::process_query(const std::string &query,
                                                const std::string &user_id,
                                                const std::string &context)
    {
        if (!is_initialized_)
        {
            return "系统未初始化";
        }

        std::string cached_response = get_from_cache(query);
        if (!cached_response.empty())
        {
            return cached_response;
        }

        // 分类查询
        auto classification = classify_query(query);

        std::string response;
        switch (classification.query_type)
        {
        case QueryClassification::EMERGENCY_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "紧急查询 detected, using RAG only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = rag_only_response(query);
            break;
        case QueryClassification::FACTUAL_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "事实性查询 detected, using RAG only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = rag_only_response(query);
            break;
        case QueryClassification::COMPLEX_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "复杂查询 detected, using hybrid response." << std::endl;
            std::cout << "===============================" << std::endl;

            response = hybrid_response(query);
            break;
        case QueryClassification::CREATIVE_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "创意查询 detected, using LLM only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = llm_only_response(query);
            break;
        default:
            std::cout << "===============================" << std::endl;
            std::cout << "未知查询类型, using adaptive response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = hybrid_response(query);
        }

        add_to_cache(query, response);

        return response;
    }

    QueryClassification EdgeLLMRAGSystem::classify_query(const std::string &query)
    {
        if (!query_classifier_)
        {
            return QueryClassification{QueryClassification::UNKNOWN_QUERY, 0.0f, "分类器未初始化", false};
        }

        return query_classifier_->classify_query(query);
    }


    void EdgeLLMRAGSystem::rag_message_worker(const std::string &rag_text)
    {
        static const std::wregex wide_delimiter(
            L"([。！？；：\n]|\\?\\s|\\!\\s|\\；|\\，|\\、|\\|)");
        const std::wstring END_MARKER = L"END";

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

        std::wstring wide_text = converter.from_bytes(rag_text) + END_MARKER;

        std::wsregex_iterator it(wide_text.begin(), wide_text.end(), wide_delimiter);
        std::wsregex_iterator end;

        int skip_counter = 0;
        size_t last_pos = 0;
        while (it != end && skip_counter < 2)
        {
            last_pos = it->position() + it->length();
            ++it;
            ++skip_counter;
        }

        while (it != end)
        {
            size_t seg_start = last_pos;
            size_t seg_end = it->position();
            last_pos = seg_end + it->length();

            std::wstring wide_segment = wide_text.substr(seg_start, seg_end - seg_start);

            wide_segment.erase(0, wide_segment.find_first_not_of(L" \t\n\r"));
            wide_segment.erase(wide_segment.find_last_not_of(L" \t\n\r") + 1);

            if (!wide_segment.empty())
            {
                // 转换回UTF-8
                auto response1 = tts_client_.request(converter.to_bytes(wide_segment));
                std::cout << "[tts -> RAG] received: " << response1 << std::endl;
                // queue.push_text(converter.to_bytes(wide_segment));
            }
            ++it;
        }

        // 处理剩余内容
        if (last_pos < wide_text.length())
        {
            std::wstring last_segment = wide_text.substr(last_pos);
            if (!last_segment.empty())
            {
                auto response1 = tts_client_.request(converter.to_bytes(last_segment));
                std::cout << "[tts -> RAG] received: " << response1 << std::endl;
                // queue.push_text(converter.to_bytes(last_segment));
            }
        }
    }

    std::string EdgeLLMRAGSystem::rag_only_response(const std::string &query, bool preload)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        py::object results = searcher.attr("search")(query, 1, 0.5);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "\nQuery: '" << query << "'\n";
        std::cout
            << "elapsed: " << std::fixed << std::setprecision(2) << ms << " ms\n";

        if (py::len(results) == 0)
        {
            std::cout << "  No results" << std::endl;
            return "No results !!!";
        }

        std::string answer;
        for (const auto &item : results)
        {
            double sim = item["similarity"].cast<double>();
            std::string text = item["text"].cast<std::string>();
            answer = item["text"].cast<std::string>();
            std::string section = item["section"].cast<std::string>();
            std::string subsection = item["subsection"].cast<std::string>();
            std::cout << "  sim=" << std::fixed << std::setprecision(4) << sim
                      << ", section=" << section
                      << (subsection.empty() ? "" : ("/" + subsection))
                      << ", text=" << text.substr(0, 100) << "...\n";
        }

        if (!preload)
        {
            rag_message_worker(answer);
        }

        // std::string llm_query = query + "<rag>" + answer;
        // std::string llm_part = llm_only_response(llm_query);
        return answer;
    }

    std::string EdgeLLMRAGSystem::llm_only_response(const std::string &query)
    {
        auto response = llm_client_.request(query);
        std::cout << "[tts -> RAG] received: " << response << std::endl;
        return response;
    }

    std::string EdgeLLMRAGSystem::hybrid_response(const std::string &query)
    {
        // 结合RAG和LLM的响应
        std::string rag_part = rag_only_response(query,true);
        if (rag_part.find("No results") != std::string::npos){
            return llm_only_response(query);
        }
        std::string llm_query = query + "<rag>" + rag_part;
        std::string llm_part = llm_only_response(llm_query);

        return llm_part;
    }

    bool EdgeLLMRAGSystem::add_to_cache(const std::string &query, const std::string &response)
    {
        if (query_cache_.size() >= 100)
        { // 限制缓存大小
            query_cache_.clear();
        }

        query_cache_[query] = response;
        return true;
    }

    std::string EdgeLLMRAGSystem::get_from_cache(const std::string &query)
    {
        auto it = query_cache_.find(query);
        if (it != query_cache_.end())
        {
            return it->second;
        }
        return "";
    }

    bool EdgeLLMRAGSystem::is_cache_valid(const std::string &query)
    {
        return query_cache_.find(query) != query_cache_.end();
    }

    bool EdgeLLMRAGSystem::preload_common_queries()
    {
        // 预加载常用查询
        std::vector<std::string> common_queries = {
            "发动机故障",
            "制动系统",
            "空调不制冷",
            "保养周期"};

        for (const auto &query : common_queries)
        {
            if (query_cache_.find(query) == query_cache_.end())
            {
                add_to_cache(query, rag_only_response(query, true));
            }
        }

        return true;
    }

} // namespace edge_llm_rag