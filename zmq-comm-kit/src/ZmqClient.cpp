#include "ZmqClient.h"

namespace zmq_component {

ZmqClient::ZmqClient(const std::string& address) {
    setupSocket(ZMQ_REQ, address);
}

void ZmqClient::sendRequest(const std::string& message) {
    zmq::message_t request(message.size());
    memcpy(request.data(), message.data(), message.size());
    if (!socket_->send(request, zmq::send_flags::none)) {
        throw ZmqCommunicationError("Send timeout");
    }
}

std::string ZmqClient::receiveResponse() {
    zmq::message_t reply;
    if (!socket_->recv(reply)) {
        throw ZmqCommunicationError("Receive timeout");
    }
    return {static_cast<char*>(reply.data()), reply.size()};
}

std::string ZmqClient::request(const std::string& message) {
    sendRequest(message);
    return receiveResponse();
}

} // namespace zmq_component