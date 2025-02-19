import os
import json
import torch
import numpy as np
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

def get_graph_embbding(data_path, dataset, ent_emb, rel_emb):

    train_json = read_json(os.path.join(data_path, '{0}_train.json'.format(dataset)))
    train_graph_emb = torch.zeros((len(train_json), ent_emb.shape[1] * 3))
    test_json = read_json(os.path.join(data_path, '{0}_test.json'.format(dataset)))
    test_graph_emb = torch.zeros((len(test_json), ent_emb.shape[1] * 3))
    for item in tqdm(train_json):
        idx = item['index']
        query = item['query']
        graph = item['graph']
        quads_embeds = []
        head_emb = ent_emb[query[0]]
        relation_emb = rel_emb[query[1]]
        tail_emb = ent_emb[query[2]]
        query_emb = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)
        quads_embeds.append(query_emb)
        for s, r, o, t in graph:
            s_emb = ent_emb[s]
            r_emb = rel_emb[r]
            o_emb = ent_emb[o]
            quads_emb = torch.cat([s_emb, r_emb, o_emb], dim=-1)
            quads_embeds.append(quads_emb)
        quads_embeds = torch.stack(quads_embeds, dim=0)
        subgraph_embedding =quads_embeds.mean(dim=0)
        train_graph_emb[idx] = subgraph_embedding

    for item in tqdm(test_json):
        idx = item['index']
        query = item['query']
        graph = item['graph']
        #计算图的embedding
        quads_embeds = []
        head_emb = ent_emb[query[0]]
        relation_emb = rel_emb[query[1]]
        tail_emb = ent_emb[query[2]]
        query_emb = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)
        quads_embeds.append(query_emb)
        for s, r, o, t in graph:
            s_emb = ent_emb[s]
            r_emb = rel_emb[r]
            o_emb = ent_emb[o]
            quads_emb = torch.cat([s_emb, r_emb, o_emb], dim=-1)
            quads_embeds.append(quads_emb)
        quads_embeds = torch.stack(quads_embeds, dim=0)
        subgraph_embedding =quads_embeds.mean(dim=0)
        test_graph_emb[idx] = subgraph_embedding

    return train_graph_emb, test_graph_emb

if __name__ == '__main__':
    dataset = 'ICEWS14'
    tkg_model = 'regcn'
    data_path = './data/{0}'.format(dataset)
    emb_path = './data/{0}/{1}'.format(dataset,tkg_model)
    ent_emb = torch.from_numpy(np.load(os.path.join(emb_path, 'entity_embedding.npy')))
    rel_emb = torch.from_numpy(np.load(os.path.join(emb_path, 'relation_embedding.npy')))
    train_graph_emb, test_graph_emb = get_graph_embbding(data_path, dataset , ent_emb, rel_emb)
    torch.save(train_graph_emb, os.path.join(emb_path, 'train_graph_emb.pt'))
    torch.save(test_graph_emb, os.path.join(emb_path, 'test_graph_emb.pt'))