import random
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

sys.path.append(".")
from datasets.collaborative_bin import CollaborativeBinaryDataset
from embedders.knowledge.rotate import RotatE
from embedders.user.graphembedding import GCNLSTM


def create_graph_from_path(path, label, num_node_features=1, device=None):
    """
    Create a graph from a path of resources.
    
    Args:
        path (list): List of resource IDs representing the path.
        
    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    edge_index = []
    for i in range(len(path) - 1):
        edge_index.append([i, i + 1])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    x = torch.tensor(path, dtype=torch.float).view(
        -1, num_node_features
    ).to(device)  # por ejemplo: [[0], [1], [2]]

    # Aquí suponemos que cada grafo tiene una etiqueta, por ejemplo, clasificación binaria
    y = torch.tensor(
        [1 if label == "Pass" else 0], dtype=torch.float
    ).to(device)  # dummy: si último nodo es par

    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    random.seed(42)
    course = "vcourse"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knowledge_embedder: RotatE = torch.load(f"states/{course}/rotate.pth")

    in_channels = knowledge_embedder.embedding_dim
    hidden_channels = 64
    lstm_hidden_dim = 32
    output_dim = 20
    class_dim = 1
    model = GCNLSTM(
        in_channels,
        hidden_channels,
        lstm_hidden_dim,
        output_dim,
        class_dim,
        device=device,
    ).to(device)

    kfolds = 1
    learning_rate = 0.001
    num_epochs = 300
    max_interactions = 35
    batch_size = 16

    dataset = CollaborativeBinaryDataset(
        f"database/data/user_grades.json", knowledge_embedder, device
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    X_all = []
    Y_all = []
    with torch.no_grad():
        for idx_anchor, positive_sample, path in loader:
            X_all.append(positive_sample.squeeze(0))
            Y_all.append(
                1 if dataset.user_grades[str(int(idx_anchor.item()))] == "Pass" else 0
            )

    X_graphs = []
    for idx, x in enumerate(X_all):
        graph = create_graph_from_path(
            x, Y_all[idx], in_channels, device
        )
        X_graphs.append(graph)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    y_preds = []
    y_trues = []

    for i in tqdm(range(len(Y_all))):
        model = model.untrained_copy()
        model.set_criterion(criterion)

        X_train = []
        Y_train = []

        for j in range(len(Y_all)):
            if j != i:
                X_train.append(X_graphs[j])
                Y_train.append(Y_all[j])

        labels = torch.tensor(Y_train)
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()

        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(Y_train),  # o más si quieres oversampling
            replacement=True,
        )

        train_loader = DataLoader(X_train, batch_size=batch_size, sampler=sampler)

        model.train()
        for j in range(num_epochs):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch).squeeze(1)

                y_true = batch.y.view(-1).float()
                loss = criterion(output, y_true)

                loss.backward()
                optimizer.step()

        model.eval()
        X_test = X_all[0][:max_interactions]
        X_test = create_graph_from_path(X_test, Y_all[i], in_channels, device)

        proba = model.classify(X_test)[0]
        pred = 1 if proba.item() >= 0.5 else 0

        y_preds.append(pred)
        y_trues.append(Y_all[i])

    # Evaluación final
    accuracy = np.mean(np.array(y_preds) == np.array(y_trues))
    print(f"\nLOOCV Accuracy: {accuracy:.4f}")

    print("F1-score (Pass):", f1_score(y_trues, y_preds, pos_label=1))
    print("AUC:", roc_auc_score(y_trues, y_preds))

    cm = confusion_matrix(y_trues, y_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_trues, y_preds, target_names=["Fail", "Pass"]))
