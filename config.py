import random

# NUMBER_OF_CLIENTS: Number of client agents
NUMBER_OF_CLIENTS = 2
client_names = ['client_agent' + str(i) for i in range(NUMBER_OF_CLIENTS)]

# LEN_PER_ITERATION: How many datapoints each client gets per iteration (starts at 0). On iteration i, each client has (i+1) * LEN_PER_ITERATION samples
LEN_PER_ITERATION = 50

# ITERATIONS: How many iterations to run simulation for
ITERATIONS = 10

# LEN_TEST: Length of test dataset. Note whole dataset length is 1797
LEN_TEST = 300

random.seed(0)
# RANDOM_SEEDS: required for reproducibility of simulation. Seeds every iteration of the training for each client
RANDOM_SEEDS = {client_name: list(random.sample(range(0, 1000000), 100)) for client_name in client_names}
