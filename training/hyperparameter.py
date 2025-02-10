import copy
import json
import sys
from typing import Dict, List
from networkx import DiGraph, all_pairs_shortest_path_length
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(".")
from database.models.db import Resource
from datasets.knowledge import KnowledgeDataset
from embedders.knowledge.transe import TransE
from embedders.knowledge.transr import TransR
from embedders.knowledge.transh import TransH
from embedders.knowledge.rotate import RotatE
from embedders.knowledge.trans import Trans
from evaluation.metrics.mrr import mean_reciprocal_rank
from evaluation.metrics.topk_accuracy import top_k_accuracy


if __name__ == "__main__":
    courses = ["fakecourse", "ecourse", "vcourse"]  # "" is runnig

    for course in courses:
        # Triples
        with open(f"./database/data/{course}/prev_graph.json", "r") as gd:
            graph_data: dict = json.load(gd)
        with open(f"./database/data/{course}/repeat_graph.json", "r") as gb:
            graph_back: dict = json.load(gb)

        num_entities = len(set(graph_data.keys()).union(set(graph_back.keys())))
        num_triples = sum(len(v) for v in graph_data.values()) + sum(
            len(v) for v in graph_back.values()
        )
        num_relations = 2

        triples = np.zeros((num_triples, 3), dtype=int)
        relations = {"is_previous_of": 0, "needs_repeat": 1}
        with open(f"./database/data/{course}/resources.json", "r") as rd:
            resources: list[Resource] = [Resource(**r) for r in json.load(rd)]
        resource_dict: Dict[int, Resource] = {
            resource.id: resource for resource in resources
        }

        prev_graph = DiGraph()
        repeat_graph = DiGraph()
        i = 0
        for k, v in graph_data.items():
            for j in v:
                triples[i] = [
                    resource_dict[int(k)].recid,
                    relations["is_previous_of"],
                    resource_dict[int(j)].recid,
                ]
                i += 1
                prev_graph.add_edge(
                    resource_dict[int(k)].recid, resource_dict[int(j)].recid
                )

        for k, v in graph_back.items():
            for j in v:
                triples[i] = [
                    resource_dict[int(k)].recid,
                    relations["needs_repeat"],
                    resource_dict[int(j)].recid,
                ]
                i += 1
                repeat_graph.add_edge(
                    resource_dict[int(k)].recid, resource_dict[int(j)].recid
                )

        # Convertir a tensores
        triples = torch.tensor(triples, dtype=torch.long)

        # Get shortest paths
        prev_paths: dict = dict(all_pairs_shortest_path_length(prev_graph))
        repeat_paths: dict = dict(all_pairs_shortest_path_length(repeat_graph))

        # Hiperparámetros
        learning_rates = [0.001, 0.01, 0.1, 0.0001, 0.00001]
        embedding_dims = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
        ]
        margins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        batch_sizes = [4, 8, 12, 16, 20, 24]

        combinations = list(
            itertools.product(learning_rates, embedding_dims, margins, batch_sizes)
        )

        # Training history
        num_epochs = 1000
        kfolds = 10
        history = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for combination in combinations:
            learning_rate, embedding_dim, margin, batch_size = combination
            project_dim = embedding_dim

            print(f"Training with {combination}")

            fold_metrics = {}
            for fold in range(kfolds):
                best_loss = float("inf")
                best_model_state_dict = None

                # Dataset
                dataset = KnowledgeDataset(
                    triples, num_entities, 1, prev_paths, repeat_paths
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Modelo
                criterion = nn.MarginRankingLoss(margin=margin)
                model = TransE(num_entities, num_relations, embedding_dim, device, criterion) # Replace for the model you want to test
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0

                    # Crear batches
                    for heads, relations, tails, neg_tails in loader:
                        # Calcular pérdida
                        loss = model.negative_sample_loss(
                            heads.to(device),
                            relations.to(device),
                            tails.to(device),
                            neg_tails.to(device),
                        )
                        total_loss += loss.item()

                        # Optimización
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if total_loss < best_loss:
                        best_loss = total_loss
                        best_model_state_dict = copy.deepcopy(model.state_dict())

                # Test mmrr and topk
                model.load_state_dict(best_model_state_dict)

                top_k_3 = 3
                top_k_5 = 5
                top_k_7 = 7
                topk3_accuracies = []
                topk5_accuracies = []
                topk7_accuracies = []
                mmrr_accuracies = []
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
                    topk_accuracy_3 = top_k_accuracy(y_true, scores, top_k_3)
                    topk_accuracy_5 = top_k_accuracy(y_true_5, scores, top_k_5)
                    topk_accuracy_7 = top_k_accuracy(y_true_7, scores, top_k_7)
                    mmrr_accuracy = mean_reciprocal_rank(y_true, scores)
                    topk3_accuracies.append(topk_accuracy_3)
                    topk5_accuracies.append(topk_accuracy_5)
                    topk7_accuracies.append(topk_accuracy_7)
                    mmrr_accuracies.append(mmrr_accuracy)
                fold_metrics[fold] = {
                    "topk3": np.mean(topk3_accuracies),
                    "topk5": np.mean(topk5_accuracies),
                    "topk7": np.mean(topk7_accuracies),
                    "mmrr": np.mean(mmrr_accuracies),
                }
                print(
                    f"Fold {fold + 1}, Top-3: {np.mean(topk3_accuracies)}, Top-5: {np.mean(topk5_accuracies)}, Top-7: {np.mean(topk7_accuracies)}, MMRR: {np.mean(mmrr_accuracies)}"
                )

            key = (
                str(combination[0])
                + "-"
                + str(combination[1])
                + "-"
                + str(combination[2])
                + "-"
                + str(combination[3])
            )
            history["".join(key)] = fold_metrics

        json.dump(history, open(f"training/hyperparameter_history_{course}.json", "w"))