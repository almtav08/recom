import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from database.orm.query_generator import QueryGenerator


class CollaborativeBinaryDataset(Dataset):
    def __init__(self, json_file, knowledge_embedder, device):
        """
        Initialize the dataset with user paths from database and grades from JSON

        Args:
            json_file (str): Path to the JSON file containing user grades
            knowledge_embedder: The knowledge embedding model to use
        """
        # Load grades
        with open(json_file, "r") as f:
            self.user_grades = json.load(f)

        # Initialize database connection through QueryGenerator
        query_gen = QueryGenerator()
        query_gen.connect()

        self.device = device

        # Get user paths from database
        self.user_paths = {}
        users = query_gen.list_users()
        for user in users:
            if str(user.id) in self.user_grades:
                self.user_paths[str(user.id)] = torch.tensor(
                    list(map(lambda x: x.id, user.resources)), dtype=torch.long
                ).to(self.device)

        # Store knowledge embedder
        self.knowledge_embedder = knowledge_embedder

        self.valid_users = list(self.user_paths.keys())

    def __len__(self):
        return len(self.user_grades)

    def __getitem__(self, idx):
        """
        Get a training sample with positive and negative paths

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (idx_anchor, anchor_path_embedding)
        """
        anchor_path = self.user_paths[str(idx)]

        # Get embeddings for both paths
        anchor_embedding = self.knowledge_embedder.embed(anchor_path).to(self.device)
        idx_anchor = idx

        return idx_anchor, anchor_embedding
