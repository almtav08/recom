# Knowledge-based Recommender System

This project implements a recommender system that leverages knowledge embeddings to represent relationships between educational resources and provide personalized recommendations.

## Knowledge Embedders

The system uses various knowledge graph embedding models to represent educational resources and their relationships. These embedders are located in the `embedders/knowledge/` directory.

### Base Class: Trans

The `Trans` class serves as the base class for all knowledge embedders, implementing common functionality:

- Entity and relation embeddings initialization
- Common interface for embedding entities and relations
- Abstract methods for forward pass, loss calculation, and scoring triples

### Available Knowledge Embedding Models

The system implements several state-of-the-art knowledge graph embedding models:

#### TransE

`TransE` is the simplest translation-based model where relationships are represented as translations in the embedding space. For a triple (head, relation, tail), the model tries to ensure that `head + relation ≈ tail`.

- Scoring: `||h + r - t||₁`
- Simple and effective for many relationship types

#### TransH

`TransH` extends TransE by projecting entities onto relation-specific hyperplanes, allowing for more complex relationships:

- Uses normal vectors for each relation to create relation-specific projection planes
- Better handles complex relations (many-to-many, one-to-many, etc.)
- Scoring: `||h_proj + r - t_proj||₁` where `h_proj` and `t_proj` are the projected head and tail entities

#### TransR

`TransR` further extends the model by introducing separate entity and relation spaces:

- Entities and relations exist in different vector spaces
- Projection matrices transform entities into relation space before scoring
- Supports more complex relationship patterns
- Requires additional projection_dim parameter

#### RotatE

`RotatE` represents relations as rotations in complex vector space:

- Uses complex-valued embeddings (implemented as concatenated real/imaginary parts)
- Relations are modeled as rotations in complex plane
- Naturally captures symmetric/antisymmetric and inverse relations
- Scoring based on complex rotation and distance measurement

### Training

The knowledge embedders are trained using `training/knowledge.py` with:

- Margin ranking loss for negative sampling
- Adam optimizer
- Training data from course structure graphs
- Negative sampling to create contrasting examples

### Evaluation

Trained models can be evaluated using `evaluation/evaluate.py` with various metrics:

- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP@k)

Results for different models and courses can be found in the `results_analysis/` directory.

## Usage

To train a knowledge embedder:

```bash
python training/knowledge.py
```

To evaluate a trained knowledge embedder:

```bash
python evaluation/evaluate.py
```

## Project Structure

- `embedders/knowledge`: Contains knowledge embedding models
- `training`: Contains training scripts
- `evaluation`: Contains evaluation scripts and metrics
- `database`: Contains database models and ORM
- `datasets`: Contains dataset classes for training
- `results_analysis/`: Contains evaluation results