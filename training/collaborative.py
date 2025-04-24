import copy
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
from embedders.knowledge.transe import TransE
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
    for _, pos_path in batch:
        p_max = max(p_max, len(pos_path))

    samples = []
    indexes = []
    for idx_anchor, pos_path in batch:
        samples.append(F.pad(pos_path, (0, 0, 0, p_max - len(pos_path)), value=0))
        indexes.append(idx_anchor)
    samples = torch.stack(samples)
    return indexes, samples

if __name__ == "__main__":

    course = "fakecourse"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knowledge_embedder: TransE = torch.load(f"states/{course}/transe.pth")

    model = UserEmbedding(100, 50, 30, device)
    # model = UserEmbeddingClassifier(100, 50, 20, device)
    # model = UserEmbeddingCross(100, 50, 20, device)
    kfolds = 1
    learning_rate = 0.001
    epochs = 200
    margin = 2.0
    batch_size = 128

    best_loss = float("inf")
    best_model_state_dict = None
    # for fold in range(kfolds):

    #     dataset = CollaborativeDataset(f"database/data/user_grades.json", knowledge_embedder, device)
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #     criterion = ContrastiveLoss(margin=margin)
    #     model = model.untrained_copy()
    #     model.set_criterion(criterion)
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #     for epoch in range(epochs):

    #         model.train()
    #         total_loss = 0

    #         for positive_sample, negative_sample in dataloader:

    #             # Calcular pérdida
    #             loss = model.negative_sample_loss(
    #                 positive_sample.to(device),
    #                 negative_sample.to(device),
    #             )
    #             total_loss += loss.item()

    #             # Optimización
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #         if total_loss < best_loss:
    #             best_loss = total_loss
    #             best_model_state_dict = copy.deepcopy(model.state_dict())

    #         print(
    #             f"Fold {fold + 1}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.8f}",
    #         )

    # if best_model_state_dict is not None:
    #     model.load_state_dict(best_model_state_dict)
    #     torch.save(
    #         model,
    #         f"states/users/userembedding.pth".lower(),
    #     )

    for fold in range(kfolds):

        # dataset = CollaborativeDataset(f"database/data/user_grades.json", knowledge_embedder, device)
        dataset = CollaborativeBinaryDataset(f"database/data/user_grades.json", knowledge_embedder, device)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # train_dataset, _, test_dataset = random_split(dataset, [train_size, 0, test_size])
        labels = [
            1 if dataset.user_grades[str(user)] == "Pass" else 0
            for user in dataset.user_paths.keys()
        ]
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        # WeighedSampler
        targets = [
            1 if train_dataset.dataset.user_grades[str(user)] == "Pass" else 0
            for user in train_dataset.indices
        ]
        class_sample_count = torch.bincount(torch.tensor(targets))
        weights = 1.0 / class_sample_count.float()
        samples_weights = torch.tensor([weights[t] for t in targets])
        sampler = WeightedRandomSampler(
            samples_weights, len(samples_weights), replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_bin, sampler=sampler)
        train_loader_uncollate = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # criterion = ContrastiveLoss(margin=margin)
        criterion = SupConLoss(temperature=0.18)
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        model = model.untrained_copy()
        model.set_criterion(criterion)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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

                embeddings = model(positive_sample.to(device))
                labels = torch.tensor(
                    [
                        1 if dataset.user_grades[str(user)] == "Pass" else 0
                        for user in idx_anchor
                    ],
                    device=device,
                )
                loss = criterion(embeddings, labels)

                # Classification Loss
                # loss = model.compute_loss(
                #     positive_sample.to(device),
                #     torch.tensor(
                #         [
                #             1 if dataset.user_grades[str(user)] == "Pass" else 0
                #             for user in idx_anchor
                #         ],
                #         device=device,
                #         dtype=torch.long,
                #     ),
                # )
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
        torch.save(
            model,
            f"states/users/userembedding.pth".lower(),
        )

        model.eval()
        with torch.no_grad():
            # predictios = []
            # labels = []
            # for idx_anchor, positive_sample in test_loader:
            #     # probs = model.classify(positive_sample.to(device))
            #     # preds = (probs > 0.5).long()
            #     preds = model.classify(positive_sample.to(device))
            #     predictios.append(preds)
            #     labels.append(1 if dataset.user_grades[str(idx_anchor.item())] == "Pass" else 0)    # 
            # # Accuracy
            # correct = sum([1 for pred, label in zip(predictios, labels) if pred == label])
            # accuracy = correct / len(labels)
            # print(f"Accuracy: {accuracy:.4f}")

            user_embeddings = {}
            for user, path in train_loader_uncollate:
                if dataset.user_grades[str(user.item())] == "Pass":
                    user_embeddings[user.item()] = model.embed(path.to(device))

            count_p, count_f = 0, 0
            for idx_anchor, pos_test in test_loader:
                if dataset.user_grades[str(idx_anchor.item())] == "Pass" and count_p == 0:
                    print(f"User {idx_anchor.item()} has passed")
                    similarity_scores = []
                    for user, embedding in user_embeddings.items():
                        similarity = nn.functional.cosine_similarity(
                            model.embed(pos_test.to(device)),
                            torch.tensor(embedding).to(device),
                            dim=1,
                        )
                        similarity_scores.append((user, similarity.item()))
                    similarity_scores.sort(key=lambda x: x[1], reverse=False)
                    print(f"Less 5 similar users for {idx_anchor.item()}:")
                    for user, score in similarity_scores[:5]:
                        print(f"User {user} with score {score:.4f}")
                    print()
                    similarity_scores.sort(key=lambda x: x[1], reverse=True)
                    print(f"More 5 similar users for {idx_anchor.item()}:")
                    for user, score in similarity_scores[:5]:
                        print(f"User {user} with score {score:.4f}")
                    print()
                    count_p += 1
                else:
                    if count_f > 0:
                        continue
                    print(f"User {idx_anchor.item()} has failed")
                    similarity_scores = []
                    for user, embedding in user_embeddings.items():
                        similarity = nn.functional.cosine_similarity(
                            model.embed(pos_test.to(device)),
                            torch.tensor(embedding).to(device),
                            dim=1,
                        )
                        similarity_scores.append((user, similarity.item()))
                    similarity_scores.sort(key=lambda x: x[1], reverse=False)

                    print(f"Less 5 similar users for {idx_anchor.item()}:")
                    for user, score in similarity_scores[:5]:
                        print(f"User {user} with score {score:.4f}")
                    print()
                    similarity_scores.sort(key=lambda x: x[1], reverse=True)
                    print(f"More 5 similar users for {idx_anchor.item()}:")
                    for user, score in similarity_scores[:5]:
                        print(f"User {user} with score {score:.4f}")
                    print()
                    count_f += 1
