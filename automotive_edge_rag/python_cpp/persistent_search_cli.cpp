#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>

#include <chrono>
namespace fs = std::filesystem;

namespace py = pybind11;

struct CliOptions {
    int top_k = 2;
    double threshold = 0.5;
    std::vector<std::string> queries;
};

static CliOptions parse_args(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--top_k" || arg == "-k") && i + 1 < argc) {
            opts.top_k = std::stoi(argv[++i]);
        } else if ((arg == "--threshold" || arg == "-t") && i + 1 < argc) {
            opts.threshold = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--top_k N] [--threshold T] [query ...]\n";
            std::exit(0);
        } else {
            opts.queries.emplace_back(std::move(arg));
        }
    }
    return opts;
}

int main(int argc, char** argv) {
    py::scoped_interpreter guard{};

    auto opts = parse_args(argc, argv);

    try {
        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path");
        path.append("python");

        py::module_ mod = py::module_::import("vehicle_vector_search");
        py::object searcher = mod.attr("VehicleVectorSearch")("vector_db");

        std::cout << "Loading model once..." << std::endl;
        auto load_t0 = std::chrono::high_resolution_clock::now();

        fs::path cpp_dir = fs::absolute(__FILE__).parent_path();
        fs::path model_path = cpp_dir.parent_path() / "models";
        searcher.attr("load_model")(model_path.string());

        auto load_t1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
        std::cout << "Model loaded (" << std::fixed << std::setprecision(2) << load_ms << " ms)" << std::endl;

        py::object stats = searcher.attr("get_statistics")();
        std::cout << "Stats: total_documents=" << stats["total_documents"].cast<int>()
                  << ", embedding_dimension=" << stats["embedding_dimension"].cast<int>()
                  << std::endl;

        auto do_search = [&](const std::string& query) {
            auto t0 = std::chrono::high_resolution_clock::now();
            py::object results = searcher.attr("search")(query, opts.top_k, opts.threshold);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::cout << "\nQuery: '" << query << "' (top_k=" << opts.top_k
                      << ", threshold=" << opts.threshold << ")\n";
            std::cout << "⏱  elapsed: " << std::fixed << std::setprecision(2) << ms << " ms\n";

            if (py::len(results) == 0) {
                std::cout << "  No results" << std::endl;
                return;
            }
            for (const auto &item : results) {
                double sim = item["similarity"].cast<double>();
                std::string text = item["text"].cast<std::string>();
                std::string section = item["section"].cast<std::string>();
                std::string subsection = item["subsection"].cast<std::string>();
                std::cout << "  sim=" << std::fixed << std::setprecision(4) << sim
                          << ", section=" << section
                          << (subsection.empty() ? "" : ("/" + subsection))
                          << ", text=" << text.substr(0, 100) << "...\n";
            }
        };

        if (!opts.queries.empty()) {
            for (const auto& q : opts.queries) do_search(q);
        } else {
            std::cout << "\nInteractive mode. Enter query (or 'quit' to exit).\n";
            std::string line;
            while (true) {
                std::cout << "> ";
                if (!std::getline(std::cin, line)) break;
                if (line == "quit" || line == "exit") break;
                if (line.empty()) continue;
                do_search(line);
            }
        }

        return 0;
    } catch (const py::error_already_set &e) {
        std::cerr << "Python异常: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }
}