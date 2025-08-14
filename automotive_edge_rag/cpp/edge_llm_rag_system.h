#pragma once

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

#include <unordered_map>
#include <atomic>
#include "query_classifier.h"
#include "ZmqServer.h"
#include "ZmqClient.h"

namespace fs = std::filesystem;

namespace py = pybind11;

namespace edge_llm_rag
{

    class EdgeLLMRAGSystem
    {
    public:
        explicit EdgeLLMRAGSystem();
        ~EdgeLLMRAGSystem();

        bool initialize();

        std::string process_query(const std::string &query,
                                  const std::string &user_id = "",
                                  const std::string &context = "");

        QueryClassification classify_query(const std::string &query);

        std::string rag_only_response(const std::string &query, bool preload = false);

        std::string llm_only_response(const std::string &query);

        std::string hybrid_response(const std::string &query);

        bool cleanup_cache();

    private:
        bool is_initialized_;

        py::object searcher;
        py::scoped_interpreter guard{};

        zmq_component::ZmqClient tts_client_{"tcp://localhost:7777"};
        zmq_component::ZmqClient llm_client_{"tcp://localhost:8899"};

        std::unique_ptr<QueryClassifier>
            query_classifier_;

        std::unordered_map<std::string, std::string> query_cache_;

        bool add_to_cache(const std::string &query, const std::string &response);
        std::string get_from_cache(const std::string &query);
        bool is_cache_valid(const std::string &query);

        void rag_message_worker(const std::string &rag_text);
        bool preload_common_queries();
    };

} // namespace edge_llm_rag