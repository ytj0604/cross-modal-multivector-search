import os
import numpy as np
import pickle as pkl
import torch
import time

from ground_truth.distance import SetwiseDistance

current_path = os.path.dirname(os.path.abspath(__file__))

class Embeddings():
    def __init__(self, model_name, split, data_name):
        self.EMBS_PATH = '/mnt/sdb1/hjyun/embs'
        self.model_name = model_name
        self.split = split
        self.data_name = data_name
        self.nreps = 5 if model_name in ['pvse', 'dive'] else 1
        
        self.img_embs, self.txt_embs = self.load_embs(reduce=True)
        assert self.img_embs.shape[1] == self.txt_embs.shape[1]
        self.cardinality = self.img_embs.shape[1]
        self.distance_fn = self.get_distance_fn(img_set_size=self.cardinality, txt_set_size=self.cardinality)
    
    def load_embs(self, reduce=True):
        '''
        reduce: delete repeated image embeddings (for pvse and dive)
        '''
        embs_path = os.path.join(self.EMBS_PATH, self.model_name)
        img_embs_path = os.path.join(embs_path, f'{self.data_name}_{self.split}_img_embs.pkl')
        txt_embs_path = os.path.join(embs_path, f'{self.data_name}_{self.split}_txt_embs.pkl')

        img_embs = pkl.load(open(img_embs_path, 'rb'))
        txt_embs = pkl.load(open(txt_embs_path, 'rb'))
        if self.model_name in ['pvse', 'dive'] and reduce:
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), self.nreps)]) 
        return img_embs, txt_embs

    def get_img_by_vector_id(self, vector_id):
        '''
        return image set id and image embedding set
        '''
        img_id = int(vector_id/self.cardinality)
        # return img_id, self.img_embs[img_id]
        return img_id

    def get_txt_by_vector_id(self, vector_id):
        '''
        return text set id and text embedding set
        '''
        txt_id = int(vector_id/self.cardinality)
        # return txt_id, self.txt_embs[txt_id]
        return txt_id

    def get_distance_fn(self, img_set_size, txt_set_size):
        if self.model_name == 'dive':
            denominator = 2; temperature = 16; temperature_txt_scale = 1.0
            distance = SetwiseDistance(img_set_size=img_set_size, txt_set_size=txt_set_size, denominator=denominator, temperature=temperature, temperature_txt_scale=temperature_txt_scale)
            distance_fn = distance.smooth_chamfer_distance
        elif self.model_name == 'pvse':
            denominator = 2; temperature = 16; temperature_txt_scale = 1.0
            distance = SetwiseDistance(img_set_size=img_set_size, txt_set_size=txt_set_size, denominator=denominator, temperature=temperature, temperature_txt_scale=temperature_txt_scale)
            distance_fn = distance.max_distance
        else:
            raise ValueError(f'Invalid model name ({self.model_name})')
        return distance_fn

class EmbeddingsWarpper():
    def __init__(self, query_type, model_name, split, data_name):
        assert query_type in ['i2t', 't2i', 'i2i', 't2t']
        self.embeddings = Embeddings(model_name=model_name, split=split, data_name=data_name)
        self.distance_fn = self.embeddings.distance_fn

        if query_type[0] == 'i':
            self.query_embs = self.embeddings.img_embs
            self.get_query_by_vector_id = self.embeddings.get_img_by_vector_id
        elif query_type[0] == 't':
            self.query_embs = self.embeddings.txt_embs
            self.get_query_by_vector_id = self.embeddings.get_txt_by_vector_id
        else:
            raise ValueError(f'Invalid query type ({query_type})')
        
        if query_type[2] == 'i':
            self.data_embs = self.embeddings.img_embs
            self.get_data_by_vector_id = self.embeddings.get_img_by_vector_id
        elif query_type[2] == 't':
            self.data_embs = self.embeddings.txt_embs
            self.get_data_by_vector_id = self.embeddings.get_txt_by_vector_id
        else:
            raise ValueError(f'Invalid query type ({query_type})')

class SingleRetrieval():
    def __init__(self, query_type, index, model_name, split, data_name, topK, nns=False):        
        if nns:
            ann_path = os.path.join('/mnt/sdb1/hjyun/embs', 'anns_results', model_name, 'nns', f'{query_type}_inds.pkl')
            
            ann_result = pkl.load(open(ann_path, 'rb'))
            self.latencies = [ 0 for _ in range(len(ann_result))]
            self.anns = ann_result
            self.distances = [[0 for _ in range(100)] for _ in range(len(ann_result))]

        else:
            ann_path = os.path.join('/mnt/sdb1/hjyun/embs', 'anns_results', model_name, 'anns', f'top{topK}', f'{query_type}_{index}.pkl')
        
            ann_result = pkl.load(open(ann_path, 'rb'))
            self.latencies = ann_result[:, 1]
            self.anns = ann_result[:, 2:topK+2]
            self.distances = ann_result[:, topK+2:]

    def retrieve(self, query_vector_id, topK):
        '''
        return topK indices, topK distances, and latency
        '''
        return self.anns[query_vector_id][:topK], self.distances[query_vector_id][:topK], self.latencies[query_vector_id]

class MultiRetriveal():
    def __init__(self, query_type, model_name, split, data_name, single_retrieval_index, embeddings, innerTopK, nns=False):
        self.embeddings = embeddings
        self.innerTopK = innerTopK
        self.single_retrieval = SingleRetrieval(query_type=query_type, index=single_retrieval_index, 
                                                model_name=model_name, split=split, data_name=data_name,
                                                nns=nns, topK=innerTopK)

    def retrieve(self, query_set_id, TopK):
        '''
        return topK indices, topK distances, and latency
        '''
        _, Q, HQ = self.embeddings.query_embs.shape
        _, D, HD = self.embeddings.data_embs.shape
        assert HQ == HD
        assert Q == D

        query_vector_ids = [query_set_id * Q + i for i in range(Q)]
        
        start_time = time.time(); duration = 0

        candidate_data_vector_ids = set()
        for qvid in query_vector_ids:
            anns, _, latency = self.single_retrieval.retrieve(qvid, self.innerTopK)
            candidate_data_vector_ids.update(anns)
            duration += latency
        candidate_data_set_ids = [self.embeddings.get_data_by_vector_id(cdvid) for cdvid in candidate_data_vector_ids]
        candidate_data_set_ids = list(set(candidate_data_set_ids))
        # print('candidate data set ids: ', candidate_data_set_ids) # DEBUG

        query_embs = self.embeddings.query_embs[query_set_id]
        query_embs = query_embs.reshape((1,) + query_embs.shape)
        query_embs = torch.tensor(query_embs, dtype=torch.float64)

        data_embss = self.embeddings.data_embs[candidate_data_set_ids]
        data_embss = torch.tensor(data_embss, dtype=torch.float64)
        sim = self.embeddings.distance_fn(query_embs.view(-1, HQ), data_embss.view(-1, HD)).flatten()

        scores_gpu, inds_gpu = sim.sort()
        inds = inds_gpu.cpu().numpy().copy()[::-1].tolist()[:TopK]
        inds = [candidate_data_set_ids[i] for i in inds] # IMPORTANT!
        scores = scores_gpu.cpu().numpy().copy()[::-1].tolist()[:TopK]

        end_time = time.time()
        duration += end_time - start_time
        return inds, scores, duration, len(candidate_data_set_ids)