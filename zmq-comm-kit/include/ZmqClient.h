#pragma once
#include "ZmqInterface.h"

namespace zmq_component {

class ZmqClient : public ZmqInterface {
public:
    explicit ZmqClient(const std::string& address = "tcp://localhost:6666");
    void sendRequest(const std::string& message);
    std::string receiveResponse();
    std::string request(const std::string& message);
};

} // namespace zmq_component