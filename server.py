from _thread import *
import threading
import socket
import json
import numpy as np
from struct import pack
import multiprocessing
from multiprocessing.pool import ThreadPool
import time

from agent import Agent
from utils import data_formatting
from message import Message
import config

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits


def client_weights_returner(inp):
    client_instance, message = inp
    client_instance.receive_weights(message)

class Server(Agent):
    def __init__(self, agent_id):
        super(Server, self).__init__(agent_id=agent_id, agent_type='server_agent')
        self.averaged_weights = {}
        self.averaged_intercepts = {}

    def initModel(self, conn, id, train_datasets, X_test, y_test):
        message = {
            "train_datasets": train_datasets[id],
            "X_test": X_test,
            "y_test": y_test,
        }

        # Message serialized
        messageString = json.dumps(message, cls=NumpyEncoder)
        # messageString = str(Message("Server", str(currentClient), messageString))
        
        #length = pack('>Q', len(messageString))
        initMessage = {
            "id": id,
            "length": len(messageString)
        }

        #conn.sendall(length)
        conn.sendall(json.dumps(initMessage).encode())
        conn.sendall(messageString.encode())
    
    def requestValues(self, conns, num_iterations):
        # active_clients = set({'client_agent1', 'client_agent2', 'client_agent0'})

        for i in range(num_iterations):
            weights = {}
            intercepts = {}

            m = multiprocessing.Manager()
            lock = m.Lock()
            # with ThreadPool(len(active_clients)) as calling_pool:
            #     args = []
            #     for client_name in active_clients:
            #         arg = {'client_name': client_name, 'iteration': i}
            #         #arg = Message(sender_name=self.name, recipient_name=client_name, body=body)
            #         args.append(arg)

            #     messages = calling_pool.map(client_computation_caller, conns, args)
            
            for conn in conns:
                message = {'iteration': i}
                conn.send(json.dumps(message).encode())

                produceWeightsMessage = json.loads(conn.recv(1024))
                print(produceWeightsMessage)
            #     id = produceWeightsMessage["id"]
            #     length = produceWeightsMessage["length"]
                
            #     messageDict = receiveMessage(conn, length)
                
            #     weights[id] = np.array(messageDict['weights'])
            #     intercepts[id] = np.array(messageDict['intercepts'])
            
            # weights_np = list(weights.values()) # the weights for this iteration!
            # intercepts_np = list(intercepts.values())
        
            # try:
            #     averaged_weights = np.average(weights_np, axis=0)  # gets rid of security offsets
            # except:
            #     raise ValueError('''DATA INSUFFICIENT: Some client does not have a sample from each class so dimension of weights is incorrect. Make
            #                         train length per iteration larger for each client to avoid this issue''')

            # averaged_intercepts = np.average(intercepts_np, axis=0)
            # self.averaged_weights[i] = averaged_weights  ## averaged weights for this iteration!!
            # self.averaged_intercepts[i] = averaged_intercepts

            # message = {
            #     "averaged_weights": self.averaged_weights,
            #     "averaged_intercepts": self.averaged_intercepts
            # }

            # # Message serialized
            # messageString = json.dumps(message, cls=NumpyEncoder)
            # # messageString = str(Message("Server", str(currentClient), messageString))
            
            # #length = pack('>Q', len(messageString))
            # averageMessage = {
            #     "length": len(messageString)
            # }
            
            # for conn in conns:
            #     conn.sendall(json.dumps(averageMessage).encode())
            #     conn.sendall(messageString.encode())

            # with ThreadPool(len(active_clients)) as returning_pool:
            #     args = []
            #     for client_name in active_clients:

            #         client_instance = self.directory.clients[client_name]
            #         body = {
            #             'iteration': i, 'return_weights': averaged_weights,
            #             'return_intercepts': averaged_intercepts
            #         }
            #         # message = Message(sender_name=self.name, recipient_name=client_name, body=body)
            #         # args.append((client_instance, message))
            #         # returning_pool.map(client_weights_returner, args)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


DISCONNECTION_MESSAGE = "TT"
SERVER = socket.gethostbyname(socket.gethostname())
PORT = 1234
ADDR = (SERVER, PORT)

conns = []

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.bind(ADDR)

def federatedLearning(server, conn, addr, data, currentClient):
    
    print(f"[NEW CONNECTION] {addr} connected.")

    isConnect = True
    server.initModel(conn=conn, id=currentClient, train_datasets=data[4], X_test=data[1], y_test=data[3])
    
    while isConnect:
        msg = conn.recv(1024).decode()
        if msg:
            if msg == DISCONNECTION_MESSAGE:
                isConnect = False
                print(f"[{addr}] has been disconnected!")
            else:
                print(f"[{addr}] {msg}")
    conn.close()

def listen(server, data):
    listener.listen()
    print(f"[LISTENING] Server is listening on {SERVER}:{PORT}")
    requestValueFlag = True

    while True:
        # Establish connection with client.
        conn, addr = listener.accept()
        conns.append(conn)
        currentClient = threading.activeCount() - 1     
        thread = threading.Thread(target=federatedLearning, args = (server, conn, addr, data, currentClient))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {currentClient + 1}")
        if (currentClient + 1 == 2 and requestValueFlag):
            requestValueFlag = False
            time.sleep(2)
            server.requestValues(conns, config.ITERATIONS)

def receiveMessage(conn, length):
    messageString = b''
    while len(messageString) < length:
        # doing it in batches is generally better than trying
        # to do it all in one go, so I believe.
        to_read = length - len(messageString)
        messageString += conn.recv(
            4096 if to_read > 4096 else to_read)
    return json.loads(messageString)

if __name__ == '__main__':

    digits = load_digits()  # using sklearn's MNIST dataset
    X, y = digits.data, digits.target

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test = X[:-config.LEN_TEST], X[-config.LEN_TEST:]
    y_train, y_test = y[:-config.LEN_TEST], y[-config.LEN_TEST:]

    # extract only amount that we require
    number_of_samples = config.NUMBER_OF_CLIENTS * config.LEN_PER_ITERATION * config.ITERATIONS

    X_train = X_train[:number_of_samples]
    y_train = y_train[:number_of_samples]

    client_ids = [i for i in range(config.NUMBER_OF_CLIENTS)]
    client_to_datasets = data_formatting.partition_data(X_train, y_train, client_ids, config.ITERATIONS, config.LEN_PER_ITERATION)
    
    data = [X_train, X_test, y_train, y_test, client_to_datasets]

    server = Server(0)

    listen(server, data)
