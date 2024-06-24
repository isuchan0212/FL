import torch.nn as nn

from utils import average_weights

class Server(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def update(self, global_state):
        self.model_state = global_state
        updated_weights = average_weights(self.model_state)
        global_state = [updated_weights for i in range(self.args.n_clients)]
