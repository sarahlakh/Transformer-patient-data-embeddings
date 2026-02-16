import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch.utils.data import Dataset, DataLoader
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = 0
CLS_TOKEN = 1

# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_sequences(sequences, max_len=100):

    all_codes = []
    for seq in sequences.values():
        for code in seq:
            if code != "NO_CODE":
                all_codes.append(code)

    encoder = LabelEncoder()
    encoder.fit(all_codes)

    encoded_sequences = []

    for seq in sequences.values():

        encoded = [CLS_TOKEN]

        for code in seq:
            if code != "NO_CODE":
                encoded.append(encoder.transform([code])[0] + 2)

        encoded = encoded[:max_len]

        if len(encoded) < max_len:
            encoded += [PAD_TOKEN] * (max_len - len(encoded))

        encoded_sequences.append(encoded)

    vocab_size = len(encoder.classes_) + 2

    return np.array(encoded_sequences, dtype=np.int64), vocab_size


# =========================================================
# DATASET
# =========================================================

class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq, dtype=torch.long)


# =========================================================
# TRANSFORMER AUTOENCODER
# =========================================================

class PatientTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=3, max_len=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(device)

        x = self.embedding(x) + self.position(positions)

        x = self.encoder(x)

        logits = self.decoder(x)

        cls_embedding = x[:, 0, :]  # embedding patient

        return logits, cls_embedding


# =========================================================
# TRAINING
# =========================================================

def train_model(model, dataloader, epochs=40):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for sequences in dataloader:

            sequences = sequences.to(device)

            optimizer.zero_grad()

            logits, _ = model(sequences)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                sequences.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# =========================================================
# EMBEDDINGS
# =========================================================

def generate_embeddings(model, data):

    model.eval()
    embeddings = []

    with torch.no_grad():
        for seq in torch.tensor(data, dtype=torch.long).to(device):
            seq = seq.unsqueeze(0)
            _, emb = model(seq)
            embeddings.append(emb.cpu().numpy()[0])

    embeddings = np.array(embeddings)

    # 🔥 normalisation = crucial pour clustering
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    return embeddings


# =========================================================
# CLUSTERING
# =========================================================

def evaluate_clustering(embeddings, n_clusters=3):

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
        algorithm="lloyd"
    )

    labels = kmeans.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, labels)
    davies = davies_bouldin_score(embeddings, labels)

    print("\n📊 RESULTATS FINAUX")
    print("Silhouette:", round(silhouette, 3))
    print("Davies-Bouldin:", round(davies, 3))

    return labels


# =========================================================
# MAIN
# =========================================================

def run_pipeline(sequences):

    print("🔹 Preparation des données...")
    data, vocab_size = prepare_sequences(sequences)

    dataset = PatientDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("🔹 Initialisation Transformer...")
    model = PatientTransformer(
        vocab_size=vocab_size,
        embed_dim=128,
        n_heads=4,
        n_layers=3,
        max_len=data.shape[1]
    ).to(device)

    print("🔹 Entraînement...")
    train_model(model, dataloader, epochs=40)

    print("🔹 Génération embeddings...")
    embeddings = generate_embeddings(model, data)

    evaluate_clustering(embeddings)

    return embeddings


# =========================================================
# EXECUTION
# =========================================================

if __name__ == "__main__":

    with open("medical_sequences_pure.pkl", "rb") as f:
        data = pickle.load(f)

    sequences = data["sequences"]

    run_pipeline(sequences)
