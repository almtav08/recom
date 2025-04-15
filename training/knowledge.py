import copy
import json
import sys
from typing import Dict, List
import matplotlib
from networkx import DiGraph, all_pairs_shortest_path_length

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(".")
from database.models.db import Resource
from database.orm.query_generator import QueryGenerator
from datasets.knowledge import KnowledgeDataset
from embedders.knowledge.transe import TransE
from embedders.knowledge.transr import TransR
from embedders.knowledge.transh import TransH
from embedders.knowledge.rotate import RotatE
from embedders.knowledge.trans import Trans


if __name__ == "__main__":
    # Triples
    course = 'fakecourse'
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
    learning_rate = 0.0001
    num_epochs = 1000
    batch_size = 20
    kfolds = 1
    embedding_dim = 120
    project_dim = 120
    margin = 2.0

    # Training history
    history = np.zeros((kfolds, num_epochs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MarginRankingLoss(margin=margin)

    models: List[Trans] = [
        # TransE(num_entities, num_relations, embedding_dim, device),
        # TransR(num_entities, num_relations, embedding_dim, project_dim, device),
        TransH(num_entities, num_relations, embedding_dim, device),
        # RotatE(num_entities, num_relations, embedding_dim, device),
    ]

    # Entrenamiento
    for basemodel in models:
        best_loss = float("inf")
        best_model_state_dict = None
        for fold in range(kfolds):
            # Dataset
            dataset = KnowledgeDataset(triples, num_entities, 1, prev_paths, repeat_paths)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Modelo
            criterion = nn.MarginRankingLoss(margin=margin)
            model = basemodel.untrained_copy()
            model.set_criterion(criterion)
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

                history[fold, epoch] = total_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model_state_dict = copy.deepcopy(model.state_dict())

                print(
                    f"Coures: {course}, Model: {model.__class__.__name__}, Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.8f}\r",
                    end="",
                )
            print()

        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
            torch.save(
                model,
                f"states/{course}/{model.__class__.__name__}.pth".lower(),
            )

        np.save(
            f"history/{course}/{model.__class__.__name__}_history.npy".lower(),
            history,
        )

        plt.figure(figsize=(10, 6))
        for i in range(history.shape[0]):
            plt.plot(history[i], label=f"Fold {i + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"history/{course}/{model.__class__.__name__}_history.png".lower())
