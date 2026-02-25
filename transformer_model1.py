import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Device: {device}")

PAD_TOKEN = 0
CLS_TOKEN = 1

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
# DATA PREPARATION
# =========================================================

def prepare_sequences(sequences, max_len=100):
    print("\n📊 Préparation...")
    
    all_codes = []
    for seq in sequences.values():
        for code in seq:
            if code != "NO_CODE":
                all_codes.append(code)
    
    encoder = LabelEncoder()
    encoder.fit(all_codes)
    
    encoded_sequences = []
    original_sequences_list = []
    
    for seq in sequences.values():
        codes = [code for code in seq if code != "NO_CODE"]
        original_sequences_list.append(codes)
        
        encoded = [CLS_TOKEN]
        for code in codes:
            encoded.append(encoder.transform([code])[0] + 2)
        
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded += [PAD_TOKEN] * (max_len - len(encoded))
        
        encoded_sequences.append(encoded)
    
    vocab_size = len(encoder.classes_) + 2
    print(f"   Vocab: {vocab_size}, Patients: {len(encoded_sequences)}")
    
    return np.array(encoded_sequences), vocab_size, encoder, original_sequences_list

# =========================================================
# AUTOENCODEUR FINAL CORRIGÉ
# =========================================================

class FinalAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, max_len=100):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        
        # CNN Encoder avec padding pour garder les dimensions
        self.conv1 = nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_len * embed_dim)
        )
        
        # Output
        self.output = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        mask = (x != PAD_TOKEN).float()
        
        # Embedding
        x = self.embedding(x) * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        # Encoder
        x = torch.relu(self.bn1(self.conv1(x)))  # (batch, 256, seq_len)
        x = torch.relu(self.bn2(self.conv2(x)))  # (batch, 128, seq_len)
        x = torch.relu(self.bn3(self.conv3(x)))  # (batch, 64, seq_len)
        
        # Attention pooling
        x = x.permute(0, 2, 1)  # (batch, seq_len, 64)
        attn_weights = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        z = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch, 64)
        
        # Decoder
        recon = self.decoder(z)
        recon = recon.view(batch_size, seq_len, self.embed_dim)
        
        # Output
        logits = self.output(recon)
        
        return z, logits

# =========================================================
# LOSS FINALE
# =========================================================

def final_loss(z, logits, target):
    batch_size = z.size(0)
    
    # 1. Reconstruction loss
    recon_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target.view(-1),
        ignore_index=PAD_TOKEN
    )
    
    # 2. Contrastive loss
    z_norm = nn.functional.normalize(z, p=2, dim=1)
    sim_matrix = torch.mm(z_norm, z_norm.T)
    mask = torch.eye(batch_size, device=z.device)
    sim_matrix = sim_matrix * (1 - mask)
    
    # Pénaliser les similarités trop élevées
    contrastive_loss = (sim_matrix ** 2).mean()
    
    # 3. Variance loss (pour éviter l'effondrement)
    variance = torch.var(z, dim=0).mean()
    variance_bonus = -0.1 * variance
    
    total_loss = recon_loss + 0.3 * contrastive_loss + variance_bonus
    
    return total_loss, {
        'recon': recon_loss.item(),
        'contrastive': contrastive_loss.item(),
        'variance': variance.item()
    }

# =========================================================
# ENTRAÎNEMENT
# =========================================================

def train_final(model, dataloader, epochs=300):
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    best_loss = float('inf')
    
    print("\n🔹 Entraînement final:")
    
    for epoch in range(epochs):
        total_loss = 0
        recon_total = 0
        cont_total = 0
        
        for sequences in dataloader:
            sequences = sequences.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            z, logits = model(sequences)
            
            # Loss
            loss, losses = final_loss(z, logits, sequences)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            recon_total += losses['recon']
            cont_total += losses['contrastive']
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = recon_total / len(dataloader)
        avg_cont = cont_total / len(dataloader)
        
        scheduler.step()
        
        # Sauvegarde
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if epoch % 30 == 0:
            print(f"   Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Cont: {avg_cont:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

# =========================================================
# ÉVALUATION
# =========================================================

def evaluate_final(model, data, original_sequences, encoder):
    print("\n🔍 Évaluation finale...")
    
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(data), 32):
            batch = torch.tensor(data[i:i+32], dtype=torch.long).to(device)
            z, _ = model(batch)
            embeddings.append(z.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    print(f"\n📊 Statistiques:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.4f} ± {embeddings.std():.4f}")
    
    # Standardisation
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # PCA
    pca = PCA(n_components=min(20, embeddings_scaled.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    print(f"   Variance expliquée: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Recherche du meilleur nombre de clusters
    best_score = -1
    best_labels = None
    best_n = 2
    all_scores = []
    
    print("\n🔍 Recherche du meilleur nombre de clusters:")
    for n in range(2, 15):
        try:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_pca)
            
            score = silhouette_score(embeddings_pca, labels)
            all_scores.append(score)
            print(f"   {n:2d} clusters: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_n = n
                best_labels = labels
        except Exception as e:
            print(f"   {n:2d} clusters: erreur")
    
    print(f"\n✅ Meilleur: {best_n} clusters (score={best_score:.4f})")
    
    # Métriques
    db_score = davies_bouldin_score(embeddings_pca, best_labels)
    ch_score = calinski_harabasz_score(embeddings_pca, best_labels)
    
    print(f"\n📈 Métriques:")
    print(f"   Davies-Bouldin: {db_score:.4f}")
    print(f"   Calinski-Harabasz: {ch_score:.0f}")
    
    # Visualisation
    fig = plt.figure(figsize=(20, 12))
    
    # t-SNE
    ax1 = fig.add_subplot(2, 3, 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings_pca)
    scatter = ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], c=best_labels, cmap='tab20', s=15, alpha=0.7)
    ax1.set_title(f't-SNE (score={best_score:.3f})')
    plt.colorbar(scatter, ax=ax1)
    
    # Distribution
    ax2 = fig.add_subplot(2, 3, 2)
    unique, counts = np.unique(best_labels, return_counts=True)
    bars = ax2.bar(unique, counts, color='skyblue', edgecolor='navy')
    ax2.set_title(f'Distribution ({best_n} clusters)')
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count}', ha='center', va='bottom')
    
    # Évolution
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(range(2, 2+len(all_scores)), all_scores, 'bo-', linewidth=2)
    ax3.axvline(x=best_n, color='red', linestyle='--')
    ax3.set_xlabel('Nombre de clusters')
    ax3.set_ylabel('Score silhouette')
    ax3.set_title('Optimisation')
    ax3.grid(True, alpha=0.3)
    
    # Codes par cluster
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    text = "CODES PAR CLUSTER\n" + "="*25 + "\n\n"
    for cluster in unique[:6]:
        idx = np.where(best_labels == cluster)[0]
        codes = []
        for i in idx[:50]:
            codes.extend(original_sequences[i][:5])
        counter = Counter(codes)
        top_codes = counter.most_common(3)
        text += f"🏥 C{cluster} ({len(idx)}):\n"
        for code, freq in top_codes:
            text += f"  • {code}: {freq}\n"
        text += "\n"
    ax4.text(0, 1, text, va='top', family='monospace', fontsize=9, transform=ax4.transAxes)
    ax4.set_title('Profils')
    
    # Heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    sample_size = min(200, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_emb = embeddings_scaled[sample_idx]
    sample_labels = best_labels[sample_idx]
    sort_idx = np.argsort(sample_labels)
    sample_emb = sample_emb[sort_idx]
    sim_matrix = np.corrcoef(sample_emb)
    im = ax5.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax5.set_title('Similarités')
    plt.colorbar(im, ax=ax5)
    
    # Métriques
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    metrics = f"MÉTRIQUES FINALES\n" + "="*15 + "\n\n"
    metrics += f"Silhouette:        {best_score:.4f}\n"
    metrics += f"Davies-Bouldin:    {db_score:.4f}\n"
    metrics += f"Calinski-Harabasz: {ch_score:.0f}\n\n"
    metrics += f"COMPOSITION:\n"
    for cluster, count in zip(unique, counts):
        metrics += f"   C{cluster}: {count} ({count/len(embeddings)*100:.1f}%)\n"
    
    ax6.text(0, 1, metrics, va='top', family='monospace', fontsize=10, transform=ax6.transAxes)
    ax6.set_title('Statistiques')
    
    plt.tight_layout()
    plt.savefig('final_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_labels, best_score

# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run_final_pipeline():
    print("="*70)
    print("🔹 AUTOENCODEUR FINAL POUR CLUSTERING MÉDICAL")
    print("="*70)
    
    # Chargement
    print("\n📂 Chargement...")
    with open("medical_sequences_pure.pkl", "rb") as f:
        data = pickle.load(f)
    
    sequences = data["sequences"]
    print(f"   Patients: {len(sequences)}")
    
    # Préparation
    data, vocab_size, encoder, original = prepare_sequences(sequences, max_len=100)
    
    # Dataset
    dataset = PatientDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Modèle
    print("\n🤖 Construction de l'autoencodeur final...")
    model = FinalAutoencoder(
        vocab_size=vocab_size,
        embed_dim=512,
        max_len=data.shape[1]
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {params:,}")
    
    # Entraînement
    train_final(model, dataloader, epochs=300)
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Évaluation
    labels, score = evaluate_final(model, data, original, encoder)
    
    print("\n" + "="*70)
    print(f"🏆 SCORE SILHOUETTE FINAL: {score:.4f}")
    print("="*70)
    
    return labels, score

# =========================================================
# EXÉCUTION
# =========================================================

if __name__ == "__main__":
    labels, score = run_final_pipeline()
