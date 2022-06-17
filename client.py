import json, ast
import socket
import numpy as np
from struct import unpack
import copy

from agent import Agent
from utils.model_evaluator import ModelEvaluator
import config

from sklearn.linear_model import SGDClassifier


class Client(Agent):
    def __init__(self, agent_id, train_datasets, X_test, y_test):
        
        super(Client, self).__init__(agent_id=agent_id, agent_type="client_agent")

        self.train_datasets = train_datasets
        self.evaluator = ModelEvaluator(X_test, y_test)

        self.computation_times = {}

        self.personal_weights = {}  # personal weights. Maps iteration (int) to weights (numpy array)
        self.personal_intercepts = {}

        self.federated_weights = {}  # averaged weights
        self.federated_intercepts = {}

        self.personal_accuracy = {}
        self.federated_accuracy = {}
 
    def produce_weights(self, message):

        body = message.body
        iteration = body['iteration']

        if iteration - 1 > len(self.train_datasets):  # iteration is indexed starting from 1
            raise (ValueError(
                'Not enough data to support a {}th iteration. Either change iteration data length in config.py or decrease amount of iterations.'.format(
                    iteration)))

        weights, intercepts = self.compute_weights(iteration)

        self.personal_weights[iteration] = weights
        self.personal_intercepts[iteration] = intercepts

        # create copies of weights and intercepts since we may be adding to them
        final_weights, final_intercepts = copy.deepcopy(weights), copy.deepcopy(intercepts)

        body = {'weights': final_weights, 'intercepts': final_intercepts, 'iter': iteration}  # generate body

        # return Message(sender_name=self.name, recipient_name=self.directory.server_agents, body=body)

    def compute_weights(self, iteration):
        # print("compute_weights")
        X, y = self.train_datasets[iteration]
        lr = SGDClassifier(alpha=0.0001, loss="log", random_state=config.RANDOM_SEEDS[self.name][iteration])
        lr.fit(X, y)
        local_weights = lr.coef_
        local_intercepts = lr.intercept_
        return local_weights, local_intercepts

def receiveMessage(socket, length):
    messageString = b''
    while len(messageString) < length:
        # doing it in batches is generally better than trying
        # to do it all in one go, so I believe.
        to_read = length - len(messageString)
        messageString += socket.recv(
            4096 if to_read > 4096 else to_read)
    return json.loads(messageString)

if __name__ == '__main__':

    socker = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    # Define the port on which you want to connect
    port = 1234

    # Connect to the server on local computer
    socker.connect(('192.168.222.128', port))

    initMessage = json.loads(socker.recv(1024))
    id = initMessage["id"]
    length = initMessage["length"]
    
    messageDict = receiveMessage(socker, length)
    
    train_datasets = messageDict["train_datasets"]
    X_test = np.asarray(messageDict["X_test"])
    y_test = np.asarray(messageDict["y_test"])
    
    client = Client(id, train_datasets, X_test, y_test)

    produceWegihtMessage = json.loads(socker.recv(1024))
    print(produceWegihtMessage)
    client.produce_weights(produceWegihtMessage)

    message = input().encode()
    socker.send(message)
    # close the connection
    socker.close()
