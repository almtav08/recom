import sys
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.append(".")
from losses.supconloss import SupConLoss
from embedders.knowledge.rotate import RotatE
from embedders.user.userembedding import UserEmbedding
from datasets.collaborative_bin import CollaborativeBinaryDataset


if __name__ == "__main__":
    random.seed(42)
    course = "vcourse"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knowledge_embedder: RotatE = torch.load(f"states/{course}/rotate.pth")

    model = UserEmbedding(100, 50, 30, device)
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
            Y_all.append(
                1 if dataset.user_grades[str(int(idx_anchor.item()))] == "Pass" else 0
            )

    criterion = SupConLoss(temperature=0.18)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.90)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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

            for k in range(
                0, max(len(positive_indices), len(negative_indices)), half_batch
            ):
                pos_batch = (
                    positive_indices[k : k + half_batch]
                    if k < len(positive_indices)
                    else []
                )
                neg_batch = (
                    negative_indices[k : k + half_batch]
                    if k < len(negative_indices)
                    else []
                )

                batch_indices = pos_batch + neg_batch
                random.shuffle(batch_indices)
                batches.append(batch_indices)

            for batch_indices in batches:
                # Forward pass
                optimizer.zero_grad()

                batch_X = [X_train[i] for i in batch_indices]
                batch_X = pad_sequence(batch_X, batch_first=True).to(device)
                batch_Y = torch.tensor(
                    [Y_train[i] for i in batch_indices], device=device
                )

                embeddings = model(batch_X)
                loss = criterion(embeddings, batch_Y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

        model.eval()

        X_test = X_all[user][:max_interactions].unsqueeze(0)
        embedding = model(X_test)[0]

        # Get embeddings for all training users
        with torch.no_grad():
            train_embeddings_pass = []
            train_embeddings_fail = []

            for i, (x_train, y_train) in enumerate(zip(X_train, Y_train)):
                x_input = x_train[:max_interactions].unsqueeze(0)
                train_embedding = model(x_input)[0]

                if y_train == 1:  # Pass
                    train_embeddings_pass.append(train_embedding)
                else:  # Fail
                    train_embeddings_fail.append(train_embedding)

            # Calculate average similarity to pass and fail groups
            if train_embeddings_pass and train_embeddings_fail:
                # Stack embeddings
                pass_embeddings = torch.stack(train_embeddings_pass)
                fail_embeddings = torch.stack(train_embeddings_fail)

                # Calculate cosine similarities
                embedding_norm = torch.nn.functional.normalize(
                    embedding.unsqueeze(0), p=2, dim=1
                )
                pass_embeddings_norm = torch.nn.functional.normalize(
                    pass_embeddings, p=2, dim=1
                )
                fail_embeddings_norm = torch.nn.functional.normalize(
                    fail_embeddings, p=2, dim=1
                )

                # Calculate mean similarity to each group
                similarity_to_pass = torch.mean(
                    torch.mm(embedding_norm, pass_embeddings_norm.t())
                )
                similarity_to_fail = torch.mean(
                    torch.mm(embedding_norm, fail_embeddings_norm.t())
                )

                # Make prediction based on higher similarity
                predicted_label = 1 if similarity_to_pass > similarity_to_fail else 0
                real_label = Y_all[user]

                y_preds.append(predicted_label)
                y_trues.append(real_label)

                # Print comparison for this user
                # print(
                #     f"User {user}: Similarity to Pass: {similarity_to_pass:.4f}, "
                #     f"Similarity to Fail: {similarity_to_fail:.4f}, "
                #     f"Predicted: {'Pass' if predicted_label == 1 else 'Fail'}, "
                #     f"Real: {'Pass' if real_label == 1 else 'Fail'}, "
                #     f"Correct: {predicted_label == real_label}"
                # )

    # Calculate and print overall accuracy
    if y_preds and y_trues:
        accuracy = sum(pred == true for pred, true in zip(y_preds, y_trues)) / len(
            y_preds
        )
        print(
            f"\nOverall Accuracy: {accuracy:.4f} ({sum(pred == true for pred, true in zip(y_preds, y_trues))}/{len(y_preds)})"
        )

        # Print confusion matrix
        tp = sum(1 for pred, true in zip(y_preds, y_trues) if pred == 1 and true == 1)
        fp = sum(1 for pred, true in zip(y_preds, y_trues) if pred == 1 and true == 0)
        tn = sum(1 for pred, true in zip(y_preds, y_trues) if pred == 0 and true == 0)
        fn = sum(1 for pred, true in zip(y_preds, y_trues) if pred == 0 and true == 1)

        print(f"\nConfusion Matrix:")
        print(f"True Positive (Pass->Pass): {tp}")
        print(f"False Positive (Fail->Pass): {fp}")
        print(f"True Negative (Fail->Fail): {tn}")
        print(f"False Negative (Pass->Fail): {fn}")

        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"Precision: {precision:.4f}")

        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"Recall: {recall:.4f}")
