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
from evaluation.metrics.mapk import average_precision_at_k


if __name__ == "__main__":
    client = QueryGenerator()
    client.connect()
    resources = client.list_resources()

    model: TransE = torch.load("states/TransE.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    course = 'fakecourse'
    # Triples
    with open(f"./database/data/{course}/prev_graph.json", "r") as gd:
        graph_data: dict = json.load(gd)
    with open(f"./database/data/{course}/repeat_graph.json", "r") as gb:
        graph_back: dict = json.load(gb)

    top_k_3 = 3
    top_k_5 = 5
    top_k_7 = 7
    mmrr_accuracies = []
    map3_scores = []
    map5_scores = []
    map7_scores = []
    for tg_entity in graph_data.keys():
        y_true = graph_data[tg_entity][:top_k_3]
        y_true_5 = graph_data[tg_entity][:top_k_5]
        y_true_7 = graph_data[tg_entity][:top_k_7]
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
        y_true_binary = [
            1 if res.id in graph_data[str(tg_entity)] else 0 for res in resources
        ]
        mmrr_accuracy = mean_reciprocal_rank(y_true, scores)
        map3 = average_precision_at_k(y_true_binary, scores.tolist(), top_k_3)
        map5 = average_precision_at_k(y_true_binary, scores.tolist(), top_k_5)
        map7 = average_precision_at_k(y_true_binary, scores.tolist(), top_k_7)
        mmrr_accuracies.append(mmrr_accuracy)
        map3_scores.append(map3)
        map5_scores.append(map5)
        map7_scores.append(map7)
    for tg_entity in graph_back.keys():
        y_true = graph_back[tg_entity][:top_k_3]
        y_true_5 = graph_back[tg_entity][:top_k_5]
        y_true_7 = graph_back[tg_entity][:top_k_7]
        tg_entity = int(tg_entity)
        scores = []
        for resource in resources:
            if resource.recid == tg_entity:
                scores.append(np.inf)
                continue
            head, relation, tail = tg_entity, 1, resource.recid
            distance = model.score(head, relation, tail)
            scores.append(distance)
        scores = torch.tensor(scores).to(device)
        y_true_binary = [
            1 if res.id in graph_back[str(tg_entity)] else 0
            for res in resources
        ]
        mmrr_accuracy = mean_reciprocal_rank(y_true, scores)
        map3 = average_precision_at_k(y_true_binary, scores.tolist(), top_k_3)
        map5 = average_precision_at_k(y_true_binary, scores.tolist(), top_k_5)
        map7 = average_precision_at_k(y_true_binary, scores.tolist(), top_k_7)
        mmrr_accuracies.append(mmrr_accuracy)
        map3_scores.append(map3)
        map5_scores.append(map5)
        map7_scores.append(map7)
    print(f"Mean MAP accuracy: {np.mean(map3_scores)}")
