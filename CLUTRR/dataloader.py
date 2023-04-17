import os
import pandas as pd
import numpy as np
import torch
import random
import jsonlines


class DataCLUTRR:
    def __init__(self, folder):
        self.r2id = dict()
        self.id2r = dict()
        self.n_predicate = 0
        self.data = dict()
        self.ans_edge_st = 0
        self.ans_edge_ed = 0
        files = os.listdir(folder)
        for file in files:
            self.data[file] = self.read_file(os.path.join(folder, file))
    
    def read_file(self, path):
        df = pd.read_csv(path, index_col = 0)
        data = list()
        story_edges = map(eval, df['story_edges'].to_list())
        edge_types = map(eval, df['edge_types'].to_list())
        query_edges = map(eval, df['query_edge'].to_list())
        targets = df['target'].to_list()
        for story_edge, edge_type, query_edge, target in zip(story_edges, edge_types, query_edges, targets):
            assert len(story_edge) == len(edge_type)
            v2id = dict()
            cur_vid = 0
            if target not in self.r2id:
                self.r2id[target] = self.n_predicate
                self.id2r[self.n_predicate] = target
                self.n_predicate += 1
            for r_type in edge_type:
                if r_type not in self.r2id:
                    self.r2id[r_type] = self.n_predicate
                    self.id2r[self.n_predicate] = r_type
                    self.n_predicate += 1
            for pair in story_edge:
                for node in pair:
                    if node not in v2id:
                        v2id[node] = cur_vid
                        cur_vid += 1
            for node in query_edge:
                assert node in v2id
            
            edge_type = list(map(lambda x:self.r2id[x], edge_type))
            story_edge = list(map(lambda x:tuple(map(lambda y:v2id[y], x)), story_edge))
            query_edge = tuple(map(lambda x:v2id[x], query_edge))
            target = self.r2id[target]
            data.append([target, story_edge, edge_type, query_edge, cur_vid])
            self.ans_edge_ed = self.n_predicate
        return data

    def get_data(self, filename):
        ret = list()
        data = self.data[filename]
        for target, story_edge, edge_type, query_edge, n_ent in data:
            A = torch.zeros((self.n_predicate, n_ent, n_ent))
            for link, e_type in zip(story_edge, edge_type):
                A[e_type,link[0],link[1]] = 1.0

            ret.append([target, query_edge, A])
        return ret