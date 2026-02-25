import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from transformer_model import MedicalTransformer

def train_manual_multihead_safe():
    
    print(" ENTRAÎNEMENT AVEC MULTIHEAD ATTENTION MANUELLE")
    print("="*60)
    
    class ManualMultiHeadTransformer(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3, max_seq_length=50):
            super().__init__()
            
            self.embed_dim = embed_dim
            self.num_heads = num_heads

            self.code_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

            self.positional_encoding = nn.Parameter(
                torch.randn(1, max_seq_length, embed_dim) * 0.1
            )

            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
            
            self.ff_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(0.1)
                ) for _ in range(num_layers)
            ])
            
            self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
            self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
            
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        def forward(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            
            x = self.code_embedding(input_ids)  # [batch, seq_len, embed_dim]
            
            x = x + self.positional_encoding[:, :seq_len, :]
            
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, embed_dim]

            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            extended_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [batch, seq_len+1]
            
            key_padding_mask = ~extended_mask.bool()
            
            for i in range(len(self.attention_layers)):
                attn_out, _ = self.attention_layers[i](
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=key_padding_mask,
                    need_weights=False
                )
                
                # Residual + norm
                x = self.norm1[i](x + attn_out)
                
                # Feed-forward
                ff_out = self.ff_layers[i](x)
                
                # Residual + norm
                x = self.norm2[i](x + ff_out)
            
            # 6. Prendre l'embedding CLS
            patient_emb = x[:, 0, :]
            
            # 7. Projection finale
            return self.output_proj(patient_emb)
    
    # 2. Dataset simplifié
    class SimpleMedicalDataset(torch.utils.data.Dataset):
        def __init__(self, data_path, max_len=50):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.sequences = data['sequences']
            self.patient_ids = list(self.sequences.keys())
            self.max_len = max_len
            
            # Vocabulaire avec tokens spéciaux
            all_codes = ['[PAD]', '[MASK]']
            for seq in self.sequences.values():
                all_codes.extend(seq[:max_len])
            
            self.encoder = LabelEncoder()
            self.encoder.fit(all_codes)
            self.vocab_size = len(self.encoder.classes_)
            
            print(f"Dataset créé:")
            print(f"  - Patients: {len(self.patient_ids)}")
            print(f"  - Vocab size: {self.vocab_size}")
            print(f"  - Max length: {max_len}")
        
        def __len__(self):
            return len(self.patient_ids)
        
        def __getitem__(self, idx):
            pid = self.patient_ids[idx]
            seq = self.sequences[pid]
            
            # Tronquer
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            
            # Encoder (+2 car [PAD]=0, [MASK]=1)
            encoded = self.encoder.transform(seq) + 2
            
            # Padding
            if len(encoded) < self.max_len:
                encoded = np.pad(encoded, (0, self.max_len - len(encoded)))
            
            # Masque
            real_len = min(len(seq), self.max_len)
            attention_mask = np.zeros(self.max_len, dtype=int)
            attention_mask[:real_len] = 1
            
            return {
                'input_ids': torch.tensor(encoded, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'patient_id': pid
            }
    
    # 3. Charger données
    print("\n📂 Chargement...")
    dataset = SimpleMedicalDataset('medical_sequences_pure.pkl', max_len=50)
    
    # 4. Créer modèle avec MultiheadAttention explicite
    print("\n🧠 Création du modèle avec MultiheadAttention manuelle...")
    model = ManualMultiHeadTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=128,
        num_heads=4,      # 4 têtes d'attention
        num_layers=3,     # 3 couches
        max_seq_length=50
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  Embed dim: 128")
    print(f"  Attention heads: 4")
    print(f"  Transformer layers: 3")
    print(f"  Utilise: torch.nn.MultiheadAttention")
    model.to(device)
    
    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux: {total_params:,}")
    
    # 5. DataLoader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'patient_id': [item['patient_id'] for item in batch]
        }
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 6. Optimizer et loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("\n🔥 Début entraînement avec MultiheadAttention...")
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_size, seq_len = input_ids.shape
            
            # MLM: masquer 15% des tokens
            targets = input_ids.clone()
            mask = torch.rand(input_ids.shape, device=device) < 0.15
            mask = mask & (input_ids != 0)
            
            masked_input = input_ids.clone()
            masked_input[mask] = 1  # [MASK]
            
            # Forward avec multi-head attention manuelle
            patient_emb = model(masked_input, attention_mask)
            
            # Projeter pour prédire tokens masqués
            patient_emb_expanded = patient_emb.unsqueeze(1).expand(-1, seq_len, -1)
            logits = torch.matmul(patient_emb_expanded, model.code_embedding.weight.T)
            
            # Loss sur tokens masqués
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            if mask_flat.sum() > 0:
                loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")
    
    # 7. Sauvegarde
    print("\n💾 Sauvegarde...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'encoder': dataset.encoder,
        'config': {
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'max_seq_length': 50
        }
    }, 'medical_manual_multihead.pth')
    
    # 8. Générer embeddings
    print("\n🧬 Génération embeddings...")
    dataloader_all = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in dataloader_all:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            patient_ids = batch['patient_id']
            
            emb = model(input_ids, attention_mask)
            
            for i, pid in enumerate(patient_ids):
                embeddings[pid] = emb[i].cpu().numpy()
    
    # Sauvegarder embeddings
    with open('medical_embeddings_manual.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'patient_ids': list(embeddings.keys())
        }, f)
    
    print(f"✅ {len(embeddings)} embeddings générés!")
    print("📁 medical_embeddings_manual.pkl")
    
    return model, dataset

if __name__ == "__main__":
    model, dataset = train_manual_multihead_safe()