#!/usr/bin/env python3

import socket, dill

HOST ="127.0.0.1"
PORT = 6060
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    connection, address = s.accept()
    with connection:
        while True:
            received_data = connection.recv(4096*2*2*2)
            if received_data:
                print("My friend at ", address, " sent me some data")
                dill.loads(received_data)
            else:
                break
        s.close()