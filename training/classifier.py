import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.append(".")
from datasets.collaborative_bin import CollaborativeBinaryDataset
from embedders.knowledge.rotate import RotatE
from embedders.user.userembeddingclass import UserEmbeddingClassifier
from embedders.user.usercross import UserEmbeddingCross


if __name__ == "__main__":
    random.seed(42)
    course = "vcourse"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knowledge_embedder: RotatE = torch.load(f"states/{course}/rotate.pth")

    model = UserEmbeddingClassifier(100, 50, 30, device)
    # model = UserEmbeddingCross(100, 50, 30, device)
    kfolds = 1
    learning_rate = 0.0001
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
            Y_all.append(1 if dataset.user_grades[str(int(idx_anchor.item()))] == "Pass" else 0)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.96)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=1e-7)

    y_preds = []
    y_trues = []

    for user in tqdm(range(len(Y_all))):
        model = model.untrained_copy()
        model.set_criterion(criterion)

        X_train = []
        Y_train = []

        for j in range(len(Y_all)):
            if j != user:
                X_train.append(X_all[j])
                Y_train.append(Y_all[j])

        model.train()
        for j in range(num_epochs):

            # Separate the indices into batches
            positive_indices = [i for i in range(len(X_train)) if Y_train[i] == 1]
            negative_indices = [i for i in range(len(X_train)) if Y_train[i] == 0]

            random.shuffle(positive_indices)
            random.shuffle(negative_indices)

            half_batch = batch_size // 2
            batches = []

            for k in range(0, max(len(positive_indices), len(negative_indices)), half_batch):
                pos_batch = positive_indices[k:k + half_batch] if k < len(positive_indices) else []
                neg_batch = negative_indices[k:k + half_batch] if k < len(negative_indices) else []

                batch_indices = pos_batch + neg_batch
                random.shuffle(batch_indices)
                batches.append(batch_indices)

            for batch_indices in batches:
                # Forward pass
                optimizer.zero_grad()

                batch_X = [X_train[i] for i in batch_indices]
                batch_X = pad_sequence(batch_X, batch_first=True).to(device)
                batch_Y = torch.tensor([Y_train[i] for i in batch_indices], device=device)

                loss = model.compute_loss(batch_X, batch_Y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

        model.eval()
        X_test = X_all[user][:max_interactions].unsqueeze(0)

        proba = model.classify(X_test)[0]
        pred = 1 if proba.item() >= 0.5 else 0
        # pred = model.classify(X_test)[0].item()

        y_preds.append(pred)
        y_trues.append(Y_all[user])

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
