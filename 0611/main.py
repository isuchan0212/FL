from model.client import Client
from model.server import Server
import torch.multiprocessing as mp
import numpy as np
from argument import parse_args
from data.loader import EllipticData
import torch

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    gpu = torch.device('cpu')
    
    # _____________
    # Load datasets
    elliptic_data = EllipticData(args)
    subgraphs = elliptic_data.get_data()
    server = Server(args)
    client = Client(args, subgraphs)

    # ____________________________
    # Federated Learning Variables
    manager = mp.Manager()
    global_state = manager.dict({i:None for i in range(args.n_clients)})
    global_train_result = manager.dict({i:{} for i in range(args.n_clients)})
    global_val_result = manager.dict({i:{} for i in range(args.n_clients)})
    global_test_result = manager.dict({i:{} for i in range(args.n_clients)})

    lock = mp.Lock()

    for cur_round in range(args.rounds):
        print('\n')
        print(f"------------------------")
        print(f"# ROUND {cur_round+1} Begin!")
        print(f"------------------------")

        np.random.seed(args.seed+cur_round)
        # _______________________________
        # Aggregating weights in a server

        if cur_round > 0 :
            print(f"# ROUND {cur_round+1} ← Update in a server...")
            for client_id in range(args.n_clients):
                client.transfer_to_server(global_state, client_id, lock)
            server.update(global_state)

        # ________________________
        # Client Processes (Train)
        print(f"# ROUND {cur_round+1} ← Train clients...")
        processes = []
        for client_id in range(args.n_clients):
            p = mp.Process(target=client.train, args=(client_id, global_state, global_train_result, cur_round, gpu, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # _______________________
        # Client Processes (Val)
        print(f"# ROUND {cur_round+1} ← Validate clients...")
        processes = []
        for client_id in range(args.n_clients):
            p = mp.Process(target=client.val, args=(client_id, global_state, global_val_result, cur_round, gpu))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


        # _______________________
        # Client Processes (Test)
        print(f"# ROUND {cur_round+1} ← Test clients...")
        processes = []
        for client_id in range(args.n_clients):
            p = mp.Process(target=client.test, args=(client_id, global_state, global_test_result, cur_round, gpu))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


        break
        # _______    
        # Results
        print(f"# ROUND {cur_round+1} ← Results")
        print(f"Train Loss: {global_train_result}")

if __name__ == '__main__':
    mp.set_start_method('spawn')  

    args, _ = parse_args()

    main(args)

