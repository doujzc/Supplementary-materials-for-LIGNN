import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import argparse

from model_ef import EdgeTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="edge_transformer")
parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--epochs',type=int, default=50)
parser.add_argument('--batch_size',type=int, default=1)
parser.add_argument('--num_message_rounds',type=int, default=8)
parser.add_argument('--dropout',type=float, default=0.2)
parser.add_argument('--dim',type=int, default=9)
parser.add_argument('--num_heads',type=int, default=1)
parser.add_argument('--max_grad_norm',type=float,default=1.0)
parser.add_argument('--share_layers', dest='share_layers', action='store_true')
parser.add_argument('--no_share_layers', dest='share_layers', action='store_false')
parser.set_defaults(share_layers=True)
parser.add_argument('--data_path',type=str,default='data_9b2173cf')
parser.add_argument('--lesion_values', action='store_true')
parser.add_argument('--lesion_scores',  action='store_true')
parser.add_argument('--update_relations', action='store_true') #this is for relation transformer
parser.add_argument('--flat_attention', action='store_true') 
parser.add_argument('--zero_init', dest='zero_init', action='store_true')  #initialization strategy for relation aware transformer
parser.add_argument('--random_init', dest='zero_init', action='store_false')
parser.set_defaults(zero_init=True)
parser.add_argument('--optimizer',type=str,default="Adam")
parser.add_argument('--scheduler',type=str,default="linear_warmup")
parser.add_argument('--num_warmup_steps',type=int,default=100)
parser.add_argument('--ff_factor',type=int,default=4)
parser.add_argument('--log_file',type=str,default='logs/clutrr_log_file.csv')
parser.add_argument('--precision',type=int,default=32,choices=[16,32])
parser.add_argument('--seed',type=int,default=42)

cl_args = parser.parse_args()
optimizer_args = {'lr':cl_args.lr}
num_training_steps = 100
scheduler_args = {'num_warmup_steps':cl_args.num_warmup_steps,'num_training_steps':num_training_steps}
cl_args.optimizer_args = optimizer_args
cl_args.scheduler_args = scheduler_args
cl_args.edge_types = 9
cl_args.target_size = 1

device = 'cpu'

def get_mrr(g, ge):
    mrr = 0.0
    h1 = 0.0
    h3 = 0.0
    h10 = 0.0
    for i in range(g, ge+1):
        mrr += 1.0 / i
        if i <= 1:
            h1 += 1
        if i <= 3:
            h3 += 1
        if i <= 10:
            h10 += 1
    return mrr / (ge-g+1), h1 / (ge-g+1), h3 / (ge-g+1), h10 / (ge-g+1)

def eval_mrr(pred_pos, pred_neg):
    y_pred_g = torch.cat([pred_pos.view(-1, 1), pred_neg], dim=1)
    y_pred_ge = torch.cat([pred_neg, pred_pos.view(-1, 1)], dim=1)
    argsort_g = torch.argsort(y_pred_g, dim=1, descending=True)
    argsort_ge = torch.argsort(y_pred_ge, dim=1, descending=True)

    ranking_list_g = torch.nonzero(argsort_g == 0, as_tuple=False)
    ranking_list_g = ranking_list_g[:, 1] + 1
    ranking_list_ge = torch.nonzero(argsort_ge == y_pred_g.shape[1]-1, as_tuple=False)
    ranking_list_ge = ranking_list_ge[:, 1] + 1

    mrr_list = torch.zeros_like(ranking_list_g, dtype=torch.float)
    hits1_list = torch.zeros_like(mrr_list)
    hits3_list = torch.zeros_like(mrr_list)
    hits10_list = torch.zeros_like(mrr_list)
    for i in range(len(mrr_list)):
        mrr_list[i], hits1_list[i], hits3_list[i], hits10_list[i] = get_mrr(ranking_list_g[i], ranking_list_ge[i])

    return {'hits@1_list': hits1_list,
            'hits@3_list': hits3_list,
            'hits@10_list': hits10_list,
            'mrr_list': mrr_list}

def main():
    with open('data/pcqm-contact/raw/train.pt', 'rb') as f:
        graphs = torch.load(f)
    with open('data/pcqm-contact/raw/test.pt', 'rb') as f:
        test_data = torch.load(f)

    # initialize
    # nrel = 9*2+3
    nrel = 9
    hidden = 10
    max_iterations = 50
    contract = 1.0
    device = torch.device('cpu')

    model = EdgeTransformer(cl_args).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    batch_size = 16
    n_epoch = 10

    n_split = (len(graphs)+batch_size-1) // batch_size

    for epoch in range(n_epoch):
        with tqdm(range(n_split), ncols=80) as _t:
            _t.set_description_str(f'Epoch: {epoch}')
            ssloss = 0.0
            for split in _t:
                batch_st = split * batch_size
                sloss = 0.0
                for x, edge_attr, edge_index, edge_label_index, edge_label in graphs[batch_st:batch_st+batch_size]:
                    # Construct A, pos, neg
                    N = x.shape[0]
                    E = edge_index.shape[1]
                    if E == 0:
                        continue

                    A = torch.zeros((N, N, nrel), device=device)
                    for i in range(N):
                        A[i,i,:] = x[i]
                    st, ed = edge_index
                    for i in range(E):
                        A[st[i],ed[i],:3] = edge_attr[i]
                    A = A.view(1, N, N, nrel)
                    batch = {}
                    batch['batched_graphs']=A
                    batch['query_edges'] = edge_label_index.T.to(device)
                    batch['query_labels'] = edge_label.to(dtype=torch.float).to(device)
                    batch['masks'] = None
                    
                    # pred = model.forward(torch.arange(N), edge_index, aux_edge_attr, N, max_iterations, contract).squeeze()
                    # pred = model.forward(batch)
                    # pred = pred[edge_label_index[0], edge_label_index[1]]
                    # print(pred.shape)

                    # loss = F.binary_cross_entropy_with_logits(pred, edge_label.type(torch.float))
                    loss = model._calculate_loss(batch)[0]

                    loss.backward()
                    sloss += loss.item()
                _t.set_postfix_str(f'loss: {sloss:.4f}')
                ssloss += sloss
                optimizer.step()
                optimizer.zero_grad()
                optimizer.zero_grad()
            print(ssloss)


    # evaluate
    torch.save(model, 'ef.pt')
    model.eval()
    stats = {}

    sloss = 0
    for x, edge_attr, edge_index, edge_label_index, edge_label in test_data:
        # Construct A, pos, neg
        N = x.shape[0]

        A = torch.zeros((N, N, nrel), device=device)
        for i in range(N):
            A[i,i,:] = x[i]
        st, ed = edge_index
        for i in range(E):
            A[st[i],ed[i],:3] = edge_attr[i]
        A = A.view(1, N, N, nrel)
        batch = {}
        batch['batched_graphs']=A
        batch['query_edges'] = edge_label_index.T
        batch['query_labels'] = edge_label.to(dtype=torch.float)
        batch['masks'] = None
        

        with torch.no_grad():
            pred = model._calculate_loss(batch)[1]

        pos_edge_index = edge_label_index[:, edge_label == 1]
        num_pos_edges = pos_edge_index.shape[1]
        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]

        if num_pos_edges > 0:
            neg_mask = torch.ones([num_pos_edges, N],
                                    dtype=torch.bool)
            neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
            # print(pred_neg, pred_neg.shape)
            mrr_list = eval_mrr(pred_pos, pred_neg)
        else:
            # Return empty stats.
            mrr_list = eval_mrr(pred_pos, pred_pos)

        for key, val in mrr_list.items():
            if key.endswith('_list'):
                key = key[:-len('_list')]
                val = float(val.mean().item())
            if np.isnan(val):
                val = 0.
            if key not in stats:
                stats[key] = [val]
            else:
                stats[key].append(val)

        
    batch_stats = {}
    for key, val in stats.items():
        mean_val = sum(val) / len(val)
        batch_stats[key] = mean_val
        # print(f"{key}: {mean_val}")
    print(batch_stats)
    print(sloss)


main()