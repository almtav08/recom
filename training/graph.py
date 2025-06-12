import copy
import random
from typing import Counter
import torch
import torch.nn as nn
import json
import sys
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
sys.path.append(".")

from embedders.user.graphembedding import GCNLSTM
from embedders.knowledge.rotate import RotatE
from database.orm.query_generator import QueryGenerator


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


query_gen = QueryGenerator()
query_gen.connect()

user_grades = {}
user_paths = {}

with open(f"database/data/user_grades.json", "r") as f:
    user_grades = json.load(f)

users = query_gen.list_users()
for user in users:
    if str(user.id) in user_grades:
        user_paths[str(user.id)] = list(map(lambda x: x.id, user.resources))

course = "vcourse"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
knowledge_embedder: RotatE = torch.load(f"states/{course}/rotate.pth")
knowemb_size = 100

dataset = []
for user_id, path in user_paths.items():
    label = user_grades[user_id]
    graph = create_graph_from_path(
        knowledge_embedder.embed(torch.tensor(path, dtype=torch.long).to(device)).to(device), label, knowemb_size, device
    )
    dataset.append(graph)

kfolds = 1
learning_rate = 0.001
epochs = 600
batch_size = 128

best_loss = float("inf")
best_model_state_dict = None

# Initialize the model
in_channels = knowemb_size
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

for fold in range(kfolds):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(dataset))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    labels = [int(data.y.item()) for data in train_dataset]
    label_counts = Counter(labels)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[int(data.y.item())] for data in train_dataset]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    model = model.untrained_copy()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.set_criterion(criterion)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch).squeeze()
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if total_loss < best_loss:
            best_loss = total_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Fold {fold + 1}, Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.8f}",
        )

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        torch.save(
            model,
            f"states/users/userembedding.pth".lower(),
        )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data).squeeze()
            pred = (out > 0.5).float()  # Umbral de 0.5 para clasificación binaria
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")

        user_embeddings = {}
        for index in train_indices:
            user_id = list(user_paths.keys())[index]
            if user_grades[user_id] == "Pass":
                user_embeddings[user_id] = model.embed(dataset[index]).to(device)

        count_p, count_f = 0, 0
        for index in test_indices:
            user_id = list(user_paths.keys())[index]
            if user_grades[user_id] == "Pass" and count_p == 0:
                print(f"User {user_id} with grade {user_grades[user_id]}")
                count_p += 1
                user_embedding = model.embed(dataset[index]).to(device)
                similarity_scores = []
                for user, embedding in user_embeddings.items():
                    similarity = torch.cosine_similarity(
                        user_embedding,
                        embedding,
                        dim=1,
                    )
                    similarity_scores.append((user, similarity.item()))
                similarity_scores.sort(key=lambda x: x[1], reverse=False)
                print(f"Less 5 similar users for {user_id}:")
                for user, score in similarity_scores[:5]:
                    print(f"User {user} with score {score:.4f}")
                print()
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                print(f"More 5 similar users for {user_id}:")
                for user, score in similarity_scores[:5]:
                    print(f"User {user} with score {score:.4f}")
                print()
            if user_grades[user_id] == "Fail" and count_f == 0:
                print(f"User {user_id} with grade {user_grades[user_id]}")
                count_f += 1
                user_embedding = model.embed(dataset[index]).to(device)
                similarity_scores = []
                for user, embedding in user_embeddings.items():
                    similarity = torch.cosine_similarity(
                        user_embedding,
                        embedding,
                        dim=1,
                    )
                    similarity_scores.append((user, similarity.item()))
                similarity_scores.sort(key=lambda x: x[1], reverse=False)
                print(f"Less 5 similar users for {user_id}:")
                for user, score in similarity_scores[:5]:
                    print(f"User {user} with score {score:.4f}")
                print()
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                print(f"More 5 similar users for {user_id}:")
                for user, score in similarity_scores[:5]:
                    print(f"User {user} with score {score:.4f}")
                print()
