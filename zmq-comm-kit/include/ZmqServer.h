#pragma once
#include "ZmqInterface.h"

namespace zmq_component
{

    class ZmqServer : public ZmqInterface
    {
    public:
        explicit ZmqServer(const std::string &address = "tcp://*:6666");
        std::string receive();
        void send(const std::string &response);
    };

} // namespace zmq_component