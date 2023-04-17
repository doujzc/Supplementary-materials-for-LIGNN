import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from model import LILayer_ty as LILayer

def generate_connectivity(N):
    edge_index = torch.stack([torch.arange(N-1), torch.arange(1, N)])
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
    Y = torch.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            Y[i,j] = 1
    return edge_index, edge_type, Y

def generate_edge_counting(N):
    edge_index = torch.stack([torch.arange(N-1), torch.arange(1, N)])
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
    Y = torch.zeros((N, N))
    for i in range(N-2):
        for j in range(i+2, N, 2):
            Y[i,j] = 1
    return edge_index, edge_type, Y


def train(in_dim, hidden_dim, generator):
    N = 10
    edge_index, edge_type, Y = generator(N)
    
    model = LILayer(in_dim, hidden_dim, 1, tol=1e-10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    max_iteration = 200
    contract = 1.0

    # train
    with tqdm(range(1000), ncols=120) as _tqdm:
        for _ in _tqdm:
            weight = Y + torch.ones_like(Y) * Y.sum() / (N*N)
            pred = model.forward(torch.arange(N), edge_index, edge_type, N,
                                max_iteration, contract).squeeze()
            loss = F.binary_cross_entropy_with_logits(pred, Y, weight, reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()
            _tqdm.set_postfix_str(f'loss: {loss.item():.4f}')

    # evaluate
    results = list()
    with torch.no_grad():
        for N in range(10, 201, 10):
            edge_index, edge_type, Y = generator(N)

            s = torch.arange(N)
            pred = model.forward(s, edge_index, edge_type, N,
                                max_iteration, contract).squeeze()
            acc = accuracy_score(Y.flatten(), pred.flatten().sigmoid() > 0.5)
            results.append(acc)
    for i in range(len(results)):
        print(f'Length = {10+10*i}, accuracy = {results[i]}')

print('Graph connectivity:')
train(1, 2, generate_connectivity)
print('Edge counting:')
train(1, 2, generate_edge_counting)