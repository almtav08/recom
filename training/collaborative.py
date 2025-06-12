from collections import defaultdict
import copy
import random
import numpy as np
from typing import List
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, hinge_loss, jaccard_score, log_loss, mean_absolute_error, roc_auc_score
import torch
import sys
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(".")
from losses.supconloss import SupConLoss
from datasets.collaborative import CollaborativeDataset
from datasets.collaborative_bin import CollaborativeBinaryDataset
from embedders.knowledge.rotate import RotatE
from embedders.user.userembedding import UserEmbedding
from embedders.user.userembeddingclass import UserEmbeddingClassifier
from embedders.user.usercross import UserEmbeddingCross
from losses.contrastive import ContrastiveLoss


def collate_fn_train(batch):
    p_max = 0
    for _, pos_path, _, neg_path in batch:
        p_max = max(p_max, len(pos_path))
        p_max = max(p_max, len(neg_path))

    samples = []
    indexes = []
    for idx_anchor, pos_path, idx_neg, neg_path in batch:
        samples.append(F.pad(pos_path, (0, 0, 0, p_max - len(pos_path)), value=0))
        samples.append(F.pad(neg_path, (0, 0, 0, p_max - len(neg_path)), value=0))
        indexes.append(idx_anchor)
        indexes.append(idx_neg)
    samples = torch.stack(samples)
    return indexes, samples


def collate_fn_bin(batch):
    p_max = 0
    for _, pos_path, _ in batch:
        p_max = max(p_max, len(pos_path))

    samples = []
    indexes = []
    for idx_anchor, pos_path, _ in batch:
        samples.append(F.pad(pos_path, (0, 0, 0, p_max - len(pos_path)), value=0))
        indexes.append(idx_anchor)
    samples = torch.stack(samples)
    return indexes, samples


@torch.jit.script
def get_topk_similar_users(
    target_embedding: torch.Tensor,
    user_emb_matrix: torch.Tensor,
    user_ids: List[int],
    topk: int = 30,
) -> List[int]:
    similarities = F.cosine_similarity(target_embedding, user_emb_matrix, dim=1)
    topk_similar_idxs = similarities.topk(topk).indices
    return [user_ids[i.item()] for i in topk_similar_idxs]


@torch.jit.script
def compute_recommendation_scores(
    path_u: torch.Tensor, last_time_id: int, num_entities: int
) -> torch.Tensor:
    scores = torch.zeros(num_entities)
    idx_matches = (path_u == last_time_id).nonzero()
    if idx_matches.size(0) > 0:
        idx = idx_matches[0][0].item()
        neig_path = path_u[idx + 1 :]
    else:
        neig_path = path_u

    for idx, item_id in enumerate(neig_path):
        score = 1.0 / (idx + 1)
        scores[int(item_id)] += score

    return scores


@torch.jit.script
def compute_fail_recommendation_scores(
    user_paths: List[torch.Tensor],
    pos_true_set: torch.Tensor,
    num_entities: int,
) -> torch.Tensor:
    scores = torch.zeros(num_entities)
    for i in range(len(user_paths)):
        neig_user_path = user_paths[i]

        for j in range(neig_user_path.size(0)):
            item_id = int(neig_user_path[j])
            already_seen = (pos_true_set == item_id).any()
            if already_seen:
                continue

            score = 1.0 / (j + 1)
            scores[item_id] += score

    return scores


if __name__ == "__main__":
    random.seed(42)
    course = "vcourse"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # knowledge_embedder: TransE = torch.load(f"states/{course}/transe.pth")
    knowledge_embedder: RotatE = torch.load(f"states/{course}/rotate.pth")

    # model = UserEmbedding(100, 50, 30, device)
    # model = UserEmbeddingClassifier(100, 50, 30, device)
    model = UserEmbeddingCross(100, 50, 30, device)
    kfolds = 1
    learning_rate = 0.001
    epochs = 600
    margin = 1.0
    batch_size = 7

    best_loss = float("inf")
    best_model_state_dict = None

    dataset = CollaborativeBinaryDataset(
        f"database/data/user_grades.json", knowledge_embedder, device
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # train_dataset, _, test_dataset = random_split(dataset, [train_size, 0, test_size])
    labels = [
        1 if dataset.user_grades[str(user)] == "Pass" else 0
        for user in dataset.user_paths.keys()
    ]

    for fold in range(kfolds):

        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=test_size, stratify=labels, random_state=42
        )

        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        # WeighedSampler
        targets = [1 if labels[user] == 1 else 0 for user in train_dataset.indices]
        class_sample_count = torch.bincount(torch.tensor(targets))
        weights = 1.0 / class_sample_count.float()
        samples_weights = torch.tensor([weights[t] for t in targets])
        sampler = WeightedRandomSampler(
            samples_weights, len(samples_weights), replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_bin,
            sampler=sampler,
        )
        train_loader_uncollate = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # criterion = ContrastiveLoss(margin=margin)
        # criterion = SupConLoss(temperature=0.18)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        model = model.untrained_copy()
        model.set_criterion(criterion)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.94)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Create a KNN model with scikit-learn
        # print("Training Model...")
        #
        # X_all = []
        # y_all = []
        # max_length = 0
        # features_list = []
        # labels_list = []
        # real_lengths = []
        #
        # for idx_anchor, positive_sample, path in train_loader_uncollate:
        #     with torch.no_grad():
        #         features = positive_sample.cpu().numpy().reshape(-1)
        #         real_lengths.append(len(features))
        #         max_length = max(max_length, len(features))
        #         features_list.append(features)
        #         labels_list.append(
        #             1 if dataset.user_grades[str(int(idx_anchor.item()))] == "Pass" else 0
        #         )
        # for idx_anchor, positive_sample, path in test_loader:
        #     with torch.no_grad():
        #         features = positive_sample.cpu().numpy().reshape(-1)
        #         max_length = max(max_length, len(features))
        #         features_list.append(features)
        #         labels_list.append(
        #             1
        #             if dataset.user_grades[str(int(idx_anchor.item()))] == "Pass"
        #             else 0
        #         )
        # for features in features_list:
        #     padded = np.zeros((max_length,))
        #     padded[:len(features)] = features
        #     X_all.append(padded)
        #
        # y_all = labels_list
        # X_all = np.array(X_all)
        # y_all = np.array(y_all)
        #
        # num_pass = np.sum(y_all == 1)
        # num_fail = np.sum(y_all == 0)
        # scale_pos_weight = num_fail / num_pass
        #
        # y_preds = []
        # y_trues = []
        #
        # max_interactions = 30 * 100
        # for i in range(len(X_all)):
        #     # Preparamos train y test para esta iteración
        #     X_train = np.delete(X_all, i, axis=0)
        #     y_train = np.delete(y_all, i, axis=0)
        #     X_test = X_all[i].reshape(1, -1)
        #     X_test[0, max_interactions:] = 0
        #     y_test = y_all[i]
        #
        #     # model = DecisionTreeClassifier(max_depth=40, random_state=42, criterion='gini', class_weight="balanced") # 60% 0.585 de umbral
        #     # model = RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=120, random_state=42, class_weight="balanced")
        #     # model = SVC(kernel="poly", C=2.0, random_state=42, max_iter=600, probability=True, class_weight="balanced")
        #     # model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        #     # model = SGDClassifier(loss="squared_hinge", penalty="l2", max_iter=600, random_state=42, class_weight="balanced") # A good one
        #     # model = SGDClassifier(loss="squared_hinge", penalty="l2", max_iter=500, random_state=42, class_weight="balanced")
        #     # model = LogisticRegression(penalty="l2", max_iter=600, random_state=42, class_weight="balanced", solver='newton-cg')
        #     model = XGBClassifier(n_estimators=70, max_depth=70, learning_rate=0.1, random_state=42, scale_pos_weight=scale_pos_weight, eval_metric=f1_score, booster="gbtree", tree_method="hist")
        #     model.fit(X_train, y_train)
        #     pred = model.predict(X_test)[0]
        #     # proba = model.predict_proba(X_test)[0][1]
        #     # pred = 1 if proba >= 0.28 else 0
        #
        #     y_preds.append(pred)
        #     y_trues.append(y_test)
        #
        # # Evaluación final
        # accuracy = np.mean(np.array(y_preds) == np.array(y_trues))
        # print(f"\nLOOCV Accuracy: {accuracy:.4f}")
        #
        # print("F1-score (Pass):", f1_score(y_trues, y_preds, pos_label=1))
        # print("AUC:", roc_auc_score(y_trues, y_preds))
        #
        # cm = confusion_matrix(y_trues, y_preds)
        # print("\nConfusion Matrix:")
        # print(cm)
        #
        # print("\nClassification Report:")
        # print(classification_report(y_trues, y_preds, target_names=["Fail", "Pass"]))
        #
        # exit()

        for epoch in range(epochs):

            model.train()
            total_loss = 0

            for idx_anchor, positive_sample in train_loader:

                optimizer.zero_grad()

                # Calcular pérdida
                # loss = model.negative_sample_loss(
                #     positive_sample.to(device),
                #     negative_sample.to(device),
                # )

                # embeddings = model(positive_sample.to(device))
                # labels = torch.tensor(
                #     [
                #         1 if dataset.user_grades[str(user)] == "Pass" else 0
                #         for user in idx_anchor
                #     ],
                #     device=device,
                # )
                # loss = criterion(embeddings, labels)

                # Classification Loss
                loss = model.compute_loss(
                    positive_sample.to(device),
                    torch.tensor(
                        [
                            1 if dataset.user_grades[str(user)] == "Pass" else 0
                            for user in idx_anchor
                        ],
                        device=device,
                        dtype=torch.long,
                    ),
                )
                total_loss += loss.item()

                # Optimización
                loss.backward()
                optimizer.step()

            if total_loss < best_loss:
                best_loss = total_loss
                best_model_state_dict = copy.deepcopy(model.state_dict())

            print(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.8f}",
            )

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

        # Save model
        torch.save(
            model,
            f"states/users/userembedding.pth".lower(),
        )

        model.eval()
        with torch.no_grad():

            print("Training Model...")

            X_all = []
            y_all = []
            max_length = 0
            features_list = []
            labels_list = []
            real_lengths = []

            for idx_anchor, positive_sample, path in train_loader_uncollate:
                with torch.no_grad():
                    features = model.embed(positive_sample.to(device))[0].cpu().numpy().reshape(-1)
                    features_list.append(positive_sample[0].cpu().numpy())
                    X_all.append(features)
                    labels_list.append(
                        1
                        if dataset.user_grades[str(int(idx_anchor.item()))] == "Pass"
                        else 0
                    )
            for idx_anchor, positive_sample, path in test_loader:
                with torch.no_grad():
                    features = model.embed(positive_sample.to(device))[0].cpu().numpy().reshape(-1)
                    features_list.append(positive_sample[0].cpu().numpy())
                    X_all.append(features)
                    labels_list.append(
                        1
                        if dataset.user_grades[str(int(idx_anchor.item()))] == "Pass"
                        else 0
                    )

            y_all = labels_list
            X_all = np.array(X_all)
            y_all = np.array(y_all)

            num_pass = np.sum(y_all == 1)
            num_fail = np.sum(y_all == 0)
            scale_pos_weight = num_fail / num_pass

            y_preds = []
            y_trues = []

            max_interactions = 30
            for i in range(len(X_all)):
                # Preparamos train y test para esta iteración
                X_train = np.delete(X_all, i, axis=0)
                y_train = np.delete(y_all, i, axis=0)
                X_test = features_list[i][:30]
                X_test = torch.tensor(X_test, device=device).unsqueeze(0)
                # X_test = model.embed(X_test)# [0].cpu().numpy().reshape(-1)
                y_test = y_all[i]

                # classifier = DecisionTreeClassifier(max_depth=40, random_state=42, criterion='gini', class_weight="balanced") # 60% 0.585 de umbral
                # classifier = RandomForestClassifier(n_estimators=50, criterion="entropy", max_depth=80, random_state=42, class_weight="balanced")
                # classifier = SVC(kernel="rbf", C=2.0, random_state=42, max_iter=100, probability=True, class_weight="balanced")
                # classifier = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
                # classifier = SGDClassifier(loss="huber", penalty="l2", max_iter=100, random_state=42, class_weight=class_weights)
                # classifier = SGDClassifier(loss="squared_hinge", penalty="l1", max_iter=100, random_state=42, class_weight=class_weights)
                # classifier = LogisticRegression(penalty="l2", max_iter=100, random_state=42, class_weight=class_weights, solver='newton-cg')
                # classifier = XGBClassifier(n_estimators=60, max_depth=60, learning_rate=0.01, random_state=42, scale_pos_weight=scale_pos_weight, eval_metric="logloss")
                # classifier.fit(X_train, y_train)
                # pred = classifier.predict([X_test])[0]
                proba = model.classify(X_test)[0]
                # proba = classifier.predict_proba([X_test])[0][1]
                print(f"Proba: {proba}, Real: {y_test}")
                pred = 1 if proba >= 0.45 else 0

                y_preds.append(pred)
                y_trues.append(y_test)

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

            exit()

            user_embeddings = {}
            user_paths = {}
            for user, path, path_b in train_loader_uncollate:
                if dataset.user_grades[str(user.item())] == "Pass":
                    user_embeddings[user.item()] = model.embed(path.to(device))[0]
                    user_paths[user.item()] = path_b[0]

            user_ids = list(user_embeddings.keys())
            user_emb_matrix = torch.stack([user_embeddings[u] for u in user_ids])  # (N, D)

            # count_p, count_f = 0, 0
            pass_correct = []
            fail_correct = []
            for idx_anchor, pos_test, path in tqdm(test_loader):
                if dataset.user_grades[str(idx_anchor.item())] == "Pass":
                    cut = int(len(pos_test[0]) * 0.5)
                    pos_true = path[:, cut:][0]
                    pos_true_test = path[:, :cut][0]
                    current_path = pos_test[:, :cut]

                    current_correct = 0
                    k = 10

                    for i in range(1, 11):
                        target_user_embedding = model.embed(current_path)
                        similar_users = get_topk_similar_users(
                            target_user_embedding, user_emb_matrix, user_ids, topk=30
                        )

                        last_time_id = pos_true_test[-1].item()
                        recommendations = defaultdict(int)
                        max_preference = 0  # Initialize a variable to keep track of the maximum preference score

                        recommendation_scores = torch.zeros(
                            knowledge_embedder.num_entities
                        )

                        for neig_user in similar_users:
                            # Calculate the resource path for the neighbor user from the last item
                            scores = compute_recommendation_scores(
                                user_paths[neig_user],
                                last_time_id,
                                knowledge_embedder.num_entities
                            )
                            recommendation_scores += scores

                        # Normalize the recommendation scores by dividing by the maximum preference score
                        max_score = recommendation_scores.max()
                        if max_score > 0:
                            recommendation_scores /= max_score

                        top_item_ids = recommendation_scores.topk(k).indices.tolist()

                        if pos_true[0].item() in top_item_ids:
                            current_correct += 1

                        current_path = pos_test[:, : cut + i]
                        pos_true_test = path[:, : cut + i][0]
                        pos_true = pos_true[1:]

                    pass_correct.append(current_correct)
                else:
                    cut = int(len(pos_test[0]) * 0.5)
                    pos_true_test = list(path[:, :cut][0])
                    visited_items_mask = torch.zeros(knowledge_embedder.num_entities, dtype=torch.bool)
                    for item in pos_true_test:
                        visited_items_mask[item.item()] = True

                    current_path = pos_test[:, :cut]
                    current_correct = 0
                    k = 10

                    for i in range(1, 11):
                        target_user_embedding = model.embed(current_path)
                        similar_users = get_topk_similar_users(
                            target_user_embedding, user_emb_matrix, user_ids, topk=30
                        )

                        neighbor_paths = [user_paths[u] for u in similar_users]
                        pos_true_tensor = torch.tensor(
                            pos_true_test, dtype=torch.long, device=device
                        )

                        recommendation_scores = compute_fail_recommendation_scores(
                            neighbor_paths,
                            pos_true_tensor,
                            knowledge_embedder.num_entities,
                        )

                        # Normalize the recommendation scores by dividing by the maximum preference score
                        max_score = recommendation_scores.max()
                        if max_score > 0:
                            recommendation_scores /= max_score

                        top_items = recommendation_scores.topk(k=10).indices

                        rand_idx = torch.randint(0, top_items.size(0), (1,), device=top_items.device).item()
                        choice = top_items[rand_idx]
                        visited_items_mask[choice] = True

                        # Extend current path with the embedding of the chosen item
                        new_embedding = knowledge_embedder.embed(
                            torch.tensor([choice], device=device)
                        ).unsqueeze(0)
                        current_path = torch.cat((current_path, new_embedding), dim=1)
                        pos_true_test.append(choice)

                        new_outcome = random.choice([0, 1])  # 1 Pass, 0 Fail
                        if new_outcome == 0:
                            current_correct += 1
                        else:
                            fail_correct.append(current_correct)
                            break

            print(fail_correct)
            print(pass_correct)
