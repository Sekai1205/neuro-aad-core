import torch
import torch.nn as nn
import torch.optim as optim

# デバイス設定 (M5 Macなら 'mps')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def main():
    print(f"--- Starting MAML Meta-Training on {device} ---")
    print("MAML Pipeline initialized successfully. Waiting for KU Leuven dataset...")

if __name__ == '__main__':
    main()