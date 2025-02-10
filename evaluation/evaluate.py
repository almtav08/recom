import json
import sys
import numpy as np
import torch

sys.path.append(".")
from database.models.db import Resource
from database.orm.query_generator import QueryGenerator
from embedders.knowledge.transe import TransE
from embedders.knowledge.transr import TransR
from embedders.knowledge.transh import TransH
from embedders.knowledge.rotate import RotatE
from evaluation.metrics.mrr import mean_reciprocal_rank
from evaluation.metrics.topk_accuracy import top_k_accuracy


if __name__ == "__main__":
    client = QueryGenerator()
    client.connect()
    resources = client.list_resources()

    model: TransE = torch.load("states/TransE.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k = 3

    course = 'fakecourse'
    # Triples
    with open(f"./database/data/{course}/prev_graph.json", "r") as gd:
        graph_data: dict = json.load(gd)
    with open(f"./database/data/{course}/repeat_graph.json", "r") as gb:
        graph_back: dict = json.load(gb)

    topk_accuracies = []
    mmrr_accuracies = []
    for tg_entity in graph_data.keys():
        y_true = graph_data[tg_entity][:top_k]
        tg_entity = int(tg_entity)
        scores = []
        for resource in resources:
            if resource.recid == tg_entity:
                scores.append(np.inf)
                continue
            head, relation, tail = tg_entity, 0, resource.recid
            distance = model.score(head, relation, tail)
            scores.append(distance)
        scores = torch.tensor(scores).to(device)
        topk_accuracy = top_k_accuracy(y_true, scores, top_k)
        mmrr_accuracy = mean_reciprocal_rank(y_true, scores)
        topk_accuracies.append(topk_accuracy)
        mmrr_accuracies.append(mmrr_accuracy)
        # print(f"Top-{top_k} accuracy for entity {tg_entity}: {topk_accuracy}")
    print(f"Mean Top-{top_k} accuracy: {np.mean(topk_accuracies)}")
    print(f"Mean MMRR accuracy: {np.mean(mmrr_accuracies)}")
