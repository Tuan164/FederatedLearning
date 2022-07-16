import json
import socket
import numpy as np
import copy
import time

from agent import Agent
from utils.model_evaluator import ModelEvaluator
import config

from sklearn.linear_model import SGDClassifier


class Client(Agent):
    def __init__(self, agent_id, conn, train_datasets, X_test, y_test):
        
        super(Client, self).__init__(agent_id=agent_id, agent_type="client_agent")

        self.conn = conn

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

        iteration = message['iteration']
        if iteration > len(self.train_datasets):  # iteration is indexed starting from 1
            raise (ValueError(
                'Not enough data to support a {}th iteration. Either change iteration data length in config.py or decrease amount of iterations.'.format(
                    iteration)))

        weights, intercepts = self.compute_weights(iteration)

        self.personal_weights[iteration] = weights
        self.personal_intercepts[iteration] = intercepts

        # create copies of weights and intercepts since we may be adding to them
        final_weights, final_intercepts = copy.deepcopy(weights), copy.deepcopy(intercepts)

        message = {'weights': final_weights, 'intercepts': final_intercepts, 'iter': iteration}

        # Message serialized
        messageString = json.dumps(message, cls=NumpyEncoder)

        produceWeightMessage = {
            "id": id,
            "length": len(messageString)
        }
        
        self.conn.sendall(json.dumps(produceWeightMessage).encode())
        self.conn.sendall(messageString.encode())

    def compute_weights(self, iteration):
        X, y = self.train_datasets[str(iteration + 1)]

        lr = SGDClassifier(alpha=0.0001, loss="log", random_state=config.RANDOM_SEEDS[self.name][iteration])

        # Assign prev round coefficients
        if iteration > 0:
            federated_weights = copy.deepcopy(self.federated_weights[iteration - 1])
            federated_intercepts = copy.deepcopy(self.federated_intercepts[iteration - 1])
        else:
            federated_weights = None
            federated_intercepts = None

        lr.fit(X, y, coef_init=federated_weights, intercept_init=federated_intercepts)
        local_weights = lr.coef_
        local_intercepts = lr.intercept_

        return local_weights, local_intercepts
    
    def receive_weights(self, message):
        iteration = message['iteration']
        averaged_weights, averaged_intercepts = message['averaged_weights'][str(iteration)], message['averaged_intercepts'][str(iteration)]
        averaged_weights = np.asarray(averaged_weights)
        averaged_intercepts = np.asarray(averaged_intercepts)

        averaged_weights = copy.deepcopy(averaged_weights)
        averaged_intercepts = copy.deepcopy(averaged_intercepts)

        self.federated_weights[iteration] = averaged_weights
        self.federated_intercepts[iteration] = averaged_intercepts

        personal_weights = self.personal_weights[iteration]
        personal_intercepts = self.personal_intercepts[iteration]

        personal_accuracy = self.evaluator.accuracy(personal_weights, personal_intercepts)
        federated_accuracy = self.evaluator.accuracy(averaged_weights, averaged_intercepts)

        self.personal_accuracy[iteration] = personal_accuracy
        self.federated_accuracy[iteration] = federated_accuracy

        args = [self.name, iteration, personal_accuracy, federated_accuracy]
        iteration_report = 'Performance Metrics for {} on iteration {} \n' \
                           '------------------------------------------- \n' \
                           'Personal accuracy: {} \n' \
                           'Federated accuracy: {} \n' \
        
        print(iteration_report.format(*args))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

    conn = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    # Define the port on which you want to connect
    port = 1234

    # Connect to the server on local computer
    conn.connect(('192.168.222.128', port))

    initMessage = json.loads(conn.recv(28))
    
    print("init: ", initMessage)
    id = initMessage["id"]
    length = initMessage["length"]
    
    messageDict = receiveMessage(conn, length)
    
    train_datasets = messageDict["train_datasets"]
    X_test = np.asarray(messageDict["X_test"])
    y_test = np.asarray(messageDict["y_test"])
    
    client = Client(id, conn, train_datasets, X_test, y_test)

    for i in range(config.ITERATIONS):
        produceWegihtMessage = json.loads(conn.recv(16))
        print(produceWegihtMessage)

        client.produce_weights(produceWegihtMessage)
        
        length = json.loads(conn.recv(18))["length"]
        averageMessage = receiveMessage(client.conn, length)
        client.receive_weights(averageMessage)

    # close the connection
    conn.close()
