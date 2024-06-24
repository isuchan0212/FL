import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    timestr = time.strftime("%m%d")

    parser.add_argument("--n_clients", type=int, default=3, choices=[3,5,10])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--local_epochs", type=int, default=1) 
    parser.add_argument("--rounds", type=int, default=100) 
    parser.add_argument("--batch_size", type=int, default=64) # 64
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    # parser.add_argument("--summary", type=str, default=timestr)
    # parser.add_argument("--final_result", type=str, default='./result.txt')

    return parser.parse_known_args()
