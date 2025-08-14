#include "ZmqInterface.h"

namespace zmq_component {

ZmqCommunicationError::ZmqCommunicationError(const std::string& what)
    : std::runtime_error("ZMQ Error: " + what) {}

void ZmqInterface::setupSocket(int socket_type, const std::string& address) {
    try {
        context_ = std::make_unique<zmq::context_t>(1);
        socket_ = std::make_unique<zmq::socket_t>(*context_, socket_type);
        
        socket_->set(zmq::sockopt::rcvtimeo, timeout_ms_);
        socket_->set(zmq::sockopt::sndtimeo, timeout_ms_);

        (socket_type == ZMQ_REP) ? socket_->bind(address) 
                                  : socket_->connect(address);
    } catch (const zmq::error_t& e) {
        throw ZmqCommunicationError(e.what());
    }
}

ZmqInterface::~ZmqInterface() {
    if (socket_) socket_->close();
    if (context_) context_->close();
}

void ZmqInterface::setTimeout(int milliseconds) {
    timeout_ms_ = milliseconds;
    if (socket_) {
        socket_->set(zmq::sockopt::rcvtimeo, timeout_ms_);
        socket_->set(zmq::sockopt::sndtimeo, timeout_ms_);
    }
}

} // namespace zmq_component