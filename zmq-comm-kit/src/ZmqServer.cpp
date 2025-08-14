#include "ZmqServer.h"

namespace zmq_component
{

    ZmqServer::ZmqServer(const std::string &address)
    {
        setupSocket(ZMQ_REP, address);
    }

    std::string ZmqServer::receive()
    {
        zmq::message_t request;
        if (!socket_->recv(request))
        {
            throw ZmqCommunicationError("Receive timeout");
        }
        return {static_cast<char *>(request.data()), request.size()};
    }

    void ZmqServer::send(const std::string &response)
    {
        zmq::message_t reply(response.size());
        memcpy(reply.data(), response.data(), response.size());
        if (!socket_->send(reply, zmq::send_flags::none))
        {
            throw ZmqCommunicationError("Send timeout");
        }
    }

} // namespace zmq_component