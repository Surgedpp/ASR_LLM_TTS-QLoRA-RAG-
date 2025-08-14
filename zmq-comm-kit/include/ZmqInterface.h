#pragma once
#include <zmq.hpp>
#include <memory>
#include <stdexcept>
#include <string>

namespace zmq_component {

class ZmqCommunicationError : public std::runtime_error {
public:
    explicit ZmqCommunicationError(const std::string& what);
};

class ZmqInterface {
protected:
    std::unique_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
    int timeout_ms_ = -1;

    void setupSocket(int socket_type, const std::string& address);
    
public:
    virtual ~ZmqInterface();
    void setTimeout(int milliseconds);
};

} // namespace zmq_component