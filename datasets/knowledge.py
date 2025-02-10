import torch
from torch.utils.data import Dataset


class KnowledgeDataset(Dataset):
    def __init__(self, triples, num_entities, min_distance, prev_paths, repeat_paths):
        super(KnowledgeDataset, self).__init__()
        self.triples = triples
        self.num_entities = num_entities
        self.min_distance = min_distance
        self.prev_paths = prev_paths
        self.repeat_paths = repeat_paths

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        paths = self.prev_paths if relation == 0 else self.repeat_paths
        neg_candidates = []
        origin_paths: dict = paths[head.item()]
        for entity in origin_paths.keys():
            if origin_paths[entity] > self.min_distance:
                neg_candidates.append(entity)
        if not neg_candidates:
            reachables = set(origin_paths.keys())
            for entity in range(self.num_entities):
                if entity not in reachables:
                    neg_candidates.append(entity)
        neg_candidates = torch.tensor(neg_candidates, dtype=torch.long)
        neg_tail_idx = torch.randint(0, len(neg_candidates), (1,))
        neg_tail = neg_candidates[neg_tail_idx][0]
        return head, relation, tail, neg_tail