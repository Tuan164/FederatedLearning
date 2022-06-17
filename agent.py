from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, agent_id, agent_type):
        self.agent_id = agent_id
        self.agent_type = agent_type  # 'CLIENT' or 'SERVER'

    @property
    def name(self):
        return str(self.agent_type) + str(self.agent_id)

