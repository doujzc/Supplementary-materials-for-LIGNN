import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import argparse

from model import LILayer_ty as LILayer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GraIL-BM_WN18RR_v1',
                    help='Name of dataset. E.g.: GraIL-BM_WN18RR_v1.')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Enable CUDA training.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-6,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=10,
                    help='Hidden dimension.')
parser.add_argument('--out_bias', action='store_true', default=False,
                    help='Add output bias on F.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size.')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='The contraction ratio gamma.')
parser.add_argument('--max_iter', type=int, default=50,
                    help='The maximum iteration steps.')
args = parser.parse_args()
device = torch.device('cuda') if args.cuda else torch.device('cpu')

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

def train(path):
    print(path)
    with open(args.dataset+'.txt', 'w') as f:
        f.write(path+'\n')

    bsz = args.batch_size
    max_iteration = args.max_iter
    contract = args.gamma
    N_epoch = args.epochs
    hidden = args.hidden

    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    r2id = dict()
    with open(os.path.join(train_path, 'relation-dic.txt'), 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            r2id[line[1]] = int(line[0])
    offset = len(r2id)

    # load train and valid data
    e2id = dict()
    edge_index = list()
    attr = list()
    train_triple = list()
    indexes = set()
    label = list()
    valid_query = list()
    with open(os.path.join(train_path, 'train.txt'), 'r') as f:
        for line in f:
            s, r, t = line.strip().split('\t')
            if s not in e2id:
                e2id[s] = len(e2id)
            if t not in e2id:
                e2id[t] = len(e2id)
            s, r, t = e2id[s], r2id[r], e2id[t]
            edge_index.append([s,t])
            edge_index.append([t,s])
            attr.append(r)
            attr.append(r+offset)
            train_triple.append((s,r,t))
            train_triple.append((t,r+offset,s))
            indexes.add((s,r,t))

    with open(os.path.join(train_path, 'valid.txt'), 'r') as f:
        for line in f:
            s, r, t = line.strip().split('\t')
            if s not in e2id or t not in e2id:
                pass
            s, r, t = e2id[s], r2id[r], e2id[t]
            indexes.add((s,r,t))
            valid_query.append([s,r,t])

    N = len(e2id)
    E = len(attr)
    edge_index = torch.tensor(edge_index).T
    edge_type = torch.tensor(attr)

    label = torch.tensor(label, dtype=torch.float)
    target_batches = list()
    valid_batches = list()
    n_batch = len(range(0, N, bsz))
    npos = torch.zeros(offset)
    for _ in range(n_batch):
        target_batches.append(list())
        valid_batches.append(list())
    for s,r,t in indexes:
        batch = s // bsz
        batch_st = batch * bsz
        target_batches[batch].append((s-batch_st,r,t))
        npos[r] += 1
    for s,r,t in valid_query:
        batch = s // bsz
        batch_st = batch * bsz
        valid_batches[batch].append((s-batch_st,r,t))
        npos[r] += 1

    mask_value = len(indexes) / (N*N*offset)

    # load test data
    e2id = dict()
    test_edge_index = list()
    attr = list()
    test_ture_triple = set()
    with open(os.path.join(test_path, 'test-graph.txt'), 'r') as f:
        for line in f:
            s, r, t = line.strip().split('\t')
            if s not in e2id:
                e2id[s] = len(e2id)
            if t not in e2id:
                e2id[t] = len(e2id)
            s, r, t = e2id[s], r2id[r], e2id[t]
            test_edge_index.append([s,t])
            test_edge_index.append([t,s])
            attr.append(r)
            attr.append(r+offset)
            test_ture_triple.add((s,r,t))
            # test_ture_triple.add((t,r+offset,s))

    test_N = len(e2id)
    test_E = len(attr)
    test_edge_index = torch.tensor(test_edge_index).T
    test_edge_type = torch.tensor(attr)

    query = list()
    with open(os.path.join(test_path, 'test-fact.txt'), 'r') as f:
        for line in f:
            s, r, t = line.strip().split('\t')
            s, r, t = e2id[s], r2id[r], e2id[t]
            query.append([s,r,t])
            # query.append([t,r+offset,s])
            test_ture_triple.add((s,r,t))
            # test_ture_triple.add((t,r+offset,s))

    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)


    test_edge_index = test_edge_index.to(device)
    test_edge_type = test_edge_type.to(device)

    # torch.manual_seed(3407)
    # torch.cuda.manual_seed(3407)
    model = LILayer(2*offset, hidden, offset, args.out_bias)
    model._reset_parameters()
    # model = torch.load(f'{args.dataset}.pt')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    


    # train
    

    s_list = torch.arange(N,device=device)


    h3_max = 0.0
    acc_max = 0.0
    auc_max = 0.0
    r_h3_max = 0.0
    valid_results = list()
    for epoch in range(N_epoch):
        sloss = 0.0
        for batch_st in range(0, N, bsz):
            batch_ed = batch_st + bsz
            
            pred = model.forward(s_list[batch_st:batch_ed], edge_index, edge_type, N, max_iteration=max_iteration, contract=contract)
            target_batch = torch.tensor(target_batches[batch_st // bsz], dtype=torch.long).T
            if target_batch.shape[0] == 0:
                continue
            target = torch.zeros((len(s_list[batch_st:batch_ed]),N,offset))
            target[target_batch[0], target_batch[2], target_batch[1]] = 1
            mask = torch.ones_like(target) * mask_value
            mask[target.type(torch.bool)] = 1
            # weight = target.clone().detach()
            # weight = weight/weight.sum(dim=[0,1]).clamp(min=1e-3) + 1/(N*bsz)
            # weight *= weight_sum

            loss = F.binary_cross_entropy_with_logits(pred, target.to(device), weight=mask.to(device), reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sloss += loss.item()

            # evaluate on valid
            true_triple = set(target_batches[batch_st // bsz])
            valid_batch = valid_batches[batch_st // bsz]
            for s,r,t in valid_batch:
                p = pred[s,:,r]
                valid_target = p[t]
                valid_mask = torch.ones(N, dtype=torch.bool)
                for tt in range(N):
                    if (s,r,tt) in true_triple:
                        valid_mask[tt] = 0
                p = p[valid_mask]
                index = torch.LongTensor(random.sample(range(len(p)), min(len(p),50)))
                p = p[index]
                g = (valid_target < p).sum()
                ge = (valid_target <= p).sum()
                valid_results.append(get_mrr(g+1, ge+1))



        # evaluate
        with torch.no_grad():
            results = list()
            r_results = list()
            rank1 = list()
            rank2 = list()

            pred = torch.zeros((test_N,test_N,offset))
            for batch_st in range(0,test_N,bsz):
                batch_ed = min(test_N, batch_st + bsz)
                pred0 = model(torch.arange(batch_st, batch_ed).to(device), test_edge_index, test_edge_type,
                                        test_N, max_iteration=max_iteration, contract=contract).cpu()
                pred[batch_st:batch_ed,:,:] = pred0
                torch.cuda.empty_cache()
            cnt = 0
            auc_pred = list()
            auc_target = list()
            # acc_threshold = list()
            for s, r, t in query:
                p = pred[s,t,:].clone()
                test_d = len(p)
                test_target = p[r].clone()
                test_mask = torch.ones(test_d,dtype=torch.bool)
                for rr in range(test_d):
                    if (s, rr, t) in test_ture_triple:
                        test_mask[rr] = 0
                    # if (t, rr, s) in test_ture_triple:
                    #     test_mask[rr+offset] = 0
                p = p[test_mask]
                index = torch.LongTensor(random.sample(range(len(p)), min(len(p),50)))
                p = p[index]
                g = (test_target < p).sum()
                rank1.append(g)
                ge = (test_target <= p).sum()
                rank2.append(ge)
                r_results.append(get_mrr(g+1, ge+1))

                p = pred[s,:,r]
                test_target = p[t]
                auc_pred.append(test_target.sigmoid().item())
                # acc_threshold.append(model.outlin.bias[r].sigmoid().item())
                auc_target.append(1)
                test_mask = torch.ones(test_N,dtype=torch.bool)
                for tt in range(test_N):
                    if (s,r,tt) in test_ture_triple:
                        test_mask[tt] = 0
                p = p[test_mask]
                index = torch.LongTensor(random.sample(range(len(p)), min(len(p),50)))
                p = p[index]
                g = (test_target < p).sum()
                rank1.append(g)
                ge = (test_target <= p).sum()
                rank2.append(ge)
                results.append(get_mrr(g+1, ge+1))
                
                # p = pred[:,t,r]
                # test_target = p[s]
                # test_mask = torch.ones(test_N,dtype=torch.bool)
                # for ss in range(test_N):
                #     if (ss,r,t) in test_ture_triple:
                #         test_mask[ss] = 0
                # p = p[test_mask]
                # index = torch.LongTensor(random.sample(range(len(p)), min(len(p),50)))
                # p = p[index]
                # g = (test_target < p).sum()
                # rank1.append(g)
                # ge = (test_target <= p).sum()
                # rank2.append(ge)
                # results.append(get_mrr(g+1, ge+1))

                # acc
                test_target = pred[s,t,r]
                ns, nr, nt = s, r, t
                while (ns, nr, nt) in test_ture_triple:
                    ns = random.randint(0, test_N-1)
                    nr = random.randint(0, test_d-1)
                    nt = random.randint(0, test_N-1)
                if nr < offset:
                    neg_target = pred[ns,nt,nr]
                else:
                    neg_target = pred[nt,ns,nr-offset]
                auc_pred.append(neg_target.sigmoid().item())
                # acc_threshold.append(model.outlin.bias[nr].sigmoid().item())
                auc_target.append(0)
                if test_target > neg_target:
                    cnt += 1

            # auc
            auc_pred = np.array(auc_pred)
            # acc_threshold = np.array(acc_threshold)
            auc_target = np.array(auc_target)
            auc_score = roc_auc_score(auc_target, auc_pred)


                
            cnt /= len(query)
            vmrr, vh1, vh3, vh10 = 0.0, 0.0, 0.0, 0.0
            for _mrr, _h1, _h3, _h10 in valid_results:
                vmrr += _mrr
                vh1 += _h1
                vh3 += _h3
                vh10 += _h10
            vmrr /= len(valid_results)
            vh1 /= len(valid_results)
            vh3 /= len(valid_results)
            vh10 /= len(valid_results)
            mrr, h1, h3, h10 = 0.0, 0.0, 0.0, 0.0
            for _mrr, _h1, _h3, _h10 in results:
                mrr += _mrr
                h1 += _h1
                h3 += _h3
                h10 += _h10
            mrr /= len(results)
            h1 /= len(results)
            h3 /= len(results)
            h10 /= len(results)

            r_mrr, r_h1, r_h3, r_h10 = 0.0, 0.0, 0.0, 0.0
            for _mrr, _h1, _h3, _h10 in r_results:
                r_mrr += _mrr
                r_h1 += _h1
                r_h3 += _h3
                r_h10 += _h10
            r_mrr /= len(r_results)
            r_h1 /= len(r_results)
            r_h3 /= len(r_results)
            r_h10 /= len(r_results)
            h3_rec = h3
            if h3 > h3_max:
                torch.save(model, f'{args.dataset}.pt')
            h3_max = max(h3_max, h3)
            r_h3_max = max(r_h3_max, r_h3)
            # acc_max = max(acc_max, cnt)
            acc_score = accuracy_score(auc_target, auc_pred>0.5)
            acc_max = max(acc_max, accuracy_score(auc_target, auc_pred>0.5))
            # acc_max = max(acc_max, accuracy_score(auc_target, auc_pred>acc_threshold))
            auc_max = max(auc_max, auc_score)
            print(f'Epoch {epoch}, loss:{sloss:.4f}, valid_h3:{vh3:.4f} h3:{h10:.4f}, rh3:{r_h3:.4f}, acc:{acc_score:.4f}, auc:{auc_score:.4f}')
            with open(args.dataset+'.txt', 'a') as f:
                f.write(f'Epoch {epoch}, loss:{sloss:.4f}, valid_h3:{vh3:.4f} h3:{h3:.4f}, rh3:{r_h3:.4f}, acc:{acc_score:.4f}, auc:{auc_score:.4f}\n')
    print('-----------')
    with open(args.dataset+'.txt', 'a') as f:
        f.write('-----------\n')

def main():
    train(os.path.join('data', args.dataset))

main()