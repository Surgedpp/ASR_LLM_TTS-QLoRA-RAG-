#pragma once

#include <Python.h>
#include <csignal>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "edge_llm_rag_system.h"
#include "ZmqServer.h"
#include "ZmqClient.h"

using namespace edge_llm_rag;

zmq_component::ZmqServer server;


void process_query(EdgeLLMRAGSystem &system, const std::string &query)
{
    std::cout << "\n 处理查询: " << query << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::string response = system.process_query(query);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n系统响应:" << std::endl;
    std::cout << response << std::endl;
    std::cout << "\n响应时间: " << duration.count() << "ms" << std::endl;
}

void exit_handler(int signal)
{
    {
        std::cout << "程序即将退出" << std::endl;
    }

    exit(signal);
}

void receive_asr_data_and_process(EdgeLLMRAGSystem &system)
{
    while (true)
    {
        std::string input_str;

        input_str = server.receive();
        std::cout << "[voice -> RAG] received: " << input_str << std::endl;
        server.send("RAG success reply !!!");
        process_query(system, input_str);
    }
}

int main()
{
    signal(SIGINT, exit_handler);

    std::cout << "初始化车载边缘LLM+RAG系统..." << std::endl;
    EdgeLLMRAGSystem system;

    if (!system.initialize())
    {
        std::cerr << "系统初始化失败" << std::endl;
        return 1;
    }

    std::cout << "系统初始化成功" << std::endl;

    receive_asr_data_and_process(system);

    return 0;
}