#include "ZmqServer.h"
#include "ZmqClient.h"
#include <iostream>
#include <thread>

int main()
{
    try
    {
        zmq_component::ZmqServer server;
        zmq_component::ZmqClient client;

        // 服务端
        std::thread server_thread([&]
                                  {
            auto request = server.receive();
            std::cout << "Server received: " << request << std::endl;
            server.send("Echo: " + request); });

        // 客户端
        auto response = client.request("Hello World!");
        std::cout << "Client received: " << response << std::endl;

        server_thread.join();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}