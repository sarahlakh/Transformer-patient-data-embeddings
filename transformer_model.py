import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# ==================== 1. TRANSFORMER STANDARD (VOTRE VERSION ACTUELLE) ====================
class PatientTransformer(nn.Module):
    """Votre version actuelle - gardez-la pour comparaison"""
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3, max_seq_length=50):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        
        x = self.embedding(input_ids)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        patient_embedding = x[:, 0, :]
        
        return patient_embedding
    

class TemporalTransformer(nn.Module):
    """Version simplifiée et corrigée du transformer temporel"""
    
    def __init__(self, code_vocab_size, temporal_vocab_size, metadata_dim=2,
                 embed_dim=96, num_heads=4, num_layers=2, max_seq_length=30):
        super().__init__()
        
        # Embeddings
        self.code_embedding = nn.Embedding(code_vocab_size, embed_dim)
        
        if temporal_vocab_size > 0:
            self.temporal_embedding = nn.Embedding(temporal_vocab_size, embed_dim // 2)
        
        if metadata_dim > 0:
            self.metadata_encoder = nn.Linear(metadata_dim, embed_dim // 2)
        
        # Transformer
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding avec +1 pour CLS
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length + 1, embed_dim) * 0.1
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection finale
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, input_ids, attention_mask, temporal_ids=None, metadata=None):
        batch_size, seq_len = input_ids.shape
        
        # 1. Embedding des codes
        x = self.code_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # 2. Ajouter temporal embedding si disponible
        if hasattr(self, 'temporal_embedding') and temporal_ids is not None:
            temporal_emb = self.temporal_embedding(temporal_ids)  # [batch, seq_len, embed_dim//2]
            # Répéter pour matcher la dimension
            temporal_emb = temporal_emb.repeat(1, 1, 2)  # [batch, seq_len, embed_dim]
            x = x + temporal_emb
        
        # 3. Ajouter metadata si disponible
        if hasattr(self, 'metadata_encoder') and metadata is not None:
            metadata_emb = self.metadata_encoder(metadata)  # [batch, embed_dim//2]
            metadata_emb = metadata_emb.repeat(1, seq_len).view(batch_size, seq_len, -1)  # [batch, seq_len, embed_dim//2]
            metadata_emb = metadata_emb.repeat(1, 1, 2)  # [batch, seq_len, embed_dim]
            x = x + metadata_emb
        
        # 4. Ajouter token CLS
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, embed_dim]
        
        # 5. Ajouter positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # 6. Ajuster masque
        cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        extended_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # 7. Transformer
        x = self.transformer(x, src_key_padding_mask=~extended_mask.bool())
        
        # 8. Embedding CLS
        patient_emb = x[:, 0, :]
        
        # 9. Projection
        return self.output_proj(patient_emb)

# ==================== 2. TRANSFORMER TEMPOREL AMÉLIORÉ ====================

class TemporalPatientDataset(Dataset):
    """Dataset pour le modèle temporel - VERSION SIMPLIFIÉE"""
    
    def __init__(self, data_path, max_seq_length=30):
        """
        Charge directement depuis le fichier pickle
        Plus besoin de passer les dicts séparément
        """
        print(f"📂 Chargement des données depuis {data_path}...")
        
        # Charger toutes les données
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extraire les composants
        self.sequences = data['sequences']
        self.temporal_features = data.get('temporal_features', {})
        self.patient_info = data.get('patient_info', {})
        
        self.patient_ids = list(self.sequences.keys())
        self.max_seq_length = max_seq_length
        
        print(f"   - {len(self.sequences)} patients chargés")
        
        # Créer encodeurs
        self._create_encoders()
        
        print(f"✅ Dataset créé: {len(self)} patients")
    
    def _create_encoders(self):
        """Crée les encodeurs pour codes et temporalité"""
        
        # 1. Tous les codes médicaux
        all_codes = []
        for seq in self.sequences.values():
            for visit in seq:
                if visit != 'NO_CODE' and visit.strip():  # Ignorer vides
                    all_codes.append(visit)
        
        if not all_codes:
            all_codes = ['NO_CODE']  # Valeur par défaut
        
        self.code_encoder = LabelEncoder()
        self.code_encoder.fit(all_codes)
        
        # 2. Toutes les features temporelles
        all_temporal = []
        for patient_id, temp_info in self.temporal_features.items():
            if 'temporal_seq' in temp_info:
                for temporal_str in temp_info['temporal_seq']:
                    if temporal_str:  # Ignorer chaînes vides
                        parts = temporal_str.split('|')
                        all_temporal.extend([p for p in parts if p])
        
        if not all_temporal:
            all_temporal = ['POS_1', 'INTERVAL_FIRST']  # Valeurs par défaut
        
        self.temporal_encoder = LabelEncoder()
        self.temporal_encoder.fit(all_temporal)
        
        print(f"   - Vocab codes: {len(self.code_encoder.classes_)}")
        print(f"   - Vocab temporel: {len(self.temporal_encoder.classes_)}")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # 1. Séquence de codes
        code_seq = self.sequences.get(patient_id, [])
        code_seq = code_seq[:self.max_seq_length]
        
        # Encoder les codes
        try:
            if len(code_seq) > 0:
                code_indices = self.code_encoder.transform(code_seq) + 1  # +1 pour padding=0
            else:
                code_indices = np.array([0])  # Séquence vide
        except ValueError as e:
            # Fallback: assigner 0 pour codes inconnus
            print(f"⚠️  Code inconnu pour patient {patient_id}, utilisation de padding")
            code_indices = np.zeros(len(code_seq), dtype=int)
        
        # 2. Séquence temporelle (si disponible)
        temporal_seq = []
        if patient_id in self.temporal_features and 'temporal_seq' in self.temporal_features[patient_id]:
            temporal_seq = self.temporal_features[patient_id]['temporal_seq'][:self.max_seq_length]
        
        temporal_indices = []
        for temp_str in temporal_seq:
            if temp_str:
                parts = temp_str.split('|')
                if parts:
                    try:
                        # Prendre la première feature comme représentative
                        temp_idx = self.temporal_encoder.transform([parts[0]])[0] + 1
                    except ValueError:
                        temp_idx = 0  # Feature temporelle inconnue
                else:
                    temp_idx = 0
            else:
                temp_idx = 0
            temporal_indices.append(temp_idx)
        
        # 3. Métadonnées
        metadata = [0.0, 0.0]  # Valeurs par défaut
        
        if patient_id in self.patient_info:
            info = self.patient_info[patient_id]
            
            # Âge (normalisé 0-1)
            age = info.get('AGE_DIAG')
            if age is not None:
                try:
                    age = float(age)
                    metadata[0] = min(age / 100.0, 1.0)
                except (ValueError, TypeError):
                    metadata[0] = 0.5  # Valeur par défaut
            
            # Pathway (encodé simple)
            pathway = info.get('Pathway')
            if pathway is not None:
                try:
                    pathway_str = str(pathway)
                    metadata[1] = (hash(pathway_str) % 10) / 10.0
                except:
                    metadata[1] = 0.0
        
        # 4. Ajuster les longueurs
        seq_len = min(len(code_seq), self.max_seq_length)
        
        # Padding pour codes
        if len(code_indices) < self.max_seq_length:
            pad_len = self.max_seq_length - len(code_indices)
            code_indices = np.pad(code_indices, (0, pad_len))
        
        # Padding pour temporal
        if len(temporal_indices) < self.max_seq_length:
            pad_len = self.max_seq_length - len(temporal_indices)
            temporal_indices = np.pad(temporal_indices, (0, pad_len))
        elif len(temporal_indices) > self.max_seq_length:
            temporal_indices = temporal_indices[:self.max_seq_length]
        
        # 5. Masque d'attention
        attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
        
        return {
            'input_ids': torch.tensor(code_indices, dtype=torch.long),
            'temporal_ids': torch.tensor(temporal_indices, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'metadata': torch.tensor(metadata, dtype=torch.float),
            'patient_id': patient_id
        }
    
class MedicalTransformer(nn.Module):
    """Transformer simplifié pour codes médicaux seulement"""
    
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3, max_seq_length=100):
        super().__init__()
        
        # Embedding des codes médicaux seulement
        self.code_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, embed_dim) * 0.1
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Token CLS pour embedding patient
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        
        # Embedding des codes
        x = self.code_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # Ajouter positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Ajouter token CLS
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, embed_dim]
        
        # Ajuster le masque pour CLS
        cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        extended_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Passer dans le transformer
        x = self.transformer(x, src_key_padding_mask=~extended_mask.bool())
        
        # Retourner l'embedding CLS (représentation patient)
        patient_embedding = x[:, 0, :]
        
        return patient_embedding
    

def load_temporal_dataset(data_path='medical_sequences_pure.pkl', max_seq_length=100):
    """Fonction simplifiée pour charger le dataset avec séquences médicales pures"""
    
    print(f"📂 Chargement du dataset depuis {data_path}...")
    
    # 1. Charger les données
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Fichier {data_path} non trouvé!")
        print("   Exécutez d'abord: python extract_medical_sequences.py")
        return None
    
    # 2. Vérifier la structure
    if 'sequences' not in data:
        print("❌ Structure de données invalide: clé 'sequences' manquante")
        return None
    
    sequences = data['sequences']
    patient_info = data.get('patient_info', {})
    
    print(f"✅ Données chargées:")
    print(f"   - Patients: {len(sequences)}")
    print(f"   - Avec infos patient: {len(patient_info)}")
    
    # 3. Créer le vocabulaire
    all_codes = []
    for seq in sequences.values():
        all_codes.extend(seq)
    
    if not all_codes:
        print("❌ Aucun code médical trouvé!")
        return None
    
    # Ajouter un token spécial pour le padding [PAD]
    all_codes = ['[PAD]'] + all_codes
    
    code_encoder = LabelEncoder()
    code_encoder.fit(all_codes)
    
    vocab_size = len(code_encoder.classes_)
    print(f"   - Taille vocabulaire: {vocab_size} (inclut [PAD])")
    
    # 4. Créer la classe Dataset interne
    class MedicalSequenceDataset(torch.utils.data.Dataset):
        def __init__(self, sequences_dict, patient_info_dict, encoder, max_len):
            self.sequences = sequences_dict
            self.patient_info = patient_info_dict
            self.patient_ids = list(sequences_dict.keys())
            self.encoder = encoder
            self.max_seq_length = max_len
            
            # Statistiques
            self.seq_lengths = [len(seq) for seq in sequences_dict.values()]
            
            print(f"   - Longueur moyenne: {np.mean(self.seq_lengths):.1f}")
            print(f"   - Longueur max: {max(self.seq_lengths)}")
            print(f"   - Longueur min: {min(self.seq_lengths)}")
        
        def __len__(self):
            return len(self.patient_ids)
        
        def __getitem__(self, idx):
            patient_id = self.patient_ids[idx]
            seq = self.sequences[patient_id]
            
            # Tronquer si trop long
            if len(seq) > self.max_seq_length:
                seq = seq[:self.max_seq_length]
            
            # Encoder les codes (0 = [PAD])
            try:
                encoded = self.encoder.transform(seq) + 1  # +1 car 0 réservé pour padding
            except ValueError as e:
                # Fallback pour codes non vus (devrait être rare)
                print(f"⚠️  Code inconnu pour {patient_id}: {e}")
                encoded = np.ones(len(seq), dtype=int)  # Valeur par défaut
            
            # Padding
            if len(encoded) < self.max_seq_length:
                pad_len = self.max_seq_length - len(encoded)
                encoded = np.pad(encoded, (0, pad_len), mode='constant')
            
            # Masque d'attention (1 pour vraies valeurs, 0 pour padding)
            seq_len = min(len(seq), self.max_seq_length)
            attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
            
            # Métadonnées (optionnelles)
            metadata = [0.0, 0.0]
            if patient_id in self.patient_info:
                info = self.patient_info[patient_id]
                
                # Âge normalisé (si disponible)
                age = info.get('AGE_DIAG')
                if age is not None:
                    try:
                        age_norm = min(float(age) / 100.0, 1.0)
                        metadata[0] = age_norm
                    except:
                        metadata[0] = 0.5
                
                # Pathway encodé (si disponible)
                pathway = info.get('Pathway')
                if pathway is not None:
                    try:
                        # Normaliser entre 0-1
                        pathway_norm = (int(pathway) % 10) / 10.0
                        metadata[1] = pathway_norm
                    except:
                        metadata[1] = 0.0
            
            return {
                'input_ids': torch.tensor(encoded, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'metadata': torch.tensor(metadata, dtype=torch.float),
                'patient_id': patient_id,
                'original_sequence': seq[:10]  # Garder les 10 premiers pour débogage
            }
    
    # 5. Créer et retourner le dataset
    dataset = MedicalSequenceDataset(
        sequences_dict=sequences,
        patient_info_dict=patient_info,
        encoder=code_encoder,
        max_len=max_seq_length
    )
    
    # 6. Afficher un exemple
    print(f"\n📝 Exemple du premier patient:")
    sample = dataset[0]
    print(f"   Patient ID: {sample['patient_id']}")
    print(f"   Sequence originale: {sample['original_sequence']}")
    print(f"   Input IDs shape: {sample['input_ids'].shape}")
    print(f"   Input IDs (premiers 10): {sample['input_ids'][:10].tolist()}")
    print(f"   Attention mask: {sample['attention_mask'][:10].tolist()}...")
    print(f"   Metadata: {sample['metadata'].tolist()}")
    
    # Stocker l'encodeur dans le dataset pour usage futur
    dataset.code_encoder = code_encoder
    dataset.vocab_size = vocab_size
    
    return dataset

# Version encore plus simple (sans metadata)
def load_dataset(data_path='medical_sequences_pure.pkl', max_seq_length=100):
    """Version ultra-simple sans metadata"""
    
    print(f"📂 Chargement simple depuis {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    patient_ids = list(sequences.keys())
    
    # Créer vocabulaire
    all_codes = ['[PAD]']
    for seq in sequences.values():
        all_codes.extend(seq)
    
    code_encoder = LabelEncoder()
    code_encoder.fit(all_codes)
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.sequences = sequences
            self.patient_ids = patient_ids
            self.encoder = code_encoder
            self.max_len = max_seq_length
            self.vocab_size = len(code_encoder.classes_)
        
        def __len__(self):
            return len(self.patient_ids)
        
        def __getitem__(self, idx):
            pid = self.patient_ids[idx]
            seq = self.sequences[pid][:self.max_len]
            
            # Encoder
            encoded = self.encoder.transform(seq) + 1
            
            # Padding
            if len(encoded) < self.max_len:
                encoded = np.pad(encoded, (0, self.max_len - len(encoded)))
            
            # Masque
            real_len = min(len(seq), self.max_len)
            mask = [1] * real_len + [0] * (self.max_len - real_len)
            
            return {
                'input_ids': torch.tensor(encoded, dtype=torch.long),
                'attention_mask': torch.tensor(mask, dtype=torch.long),
                'patient_id': pid
            }
    
    dataset = SimpleDataset()
    print(f"✅ Dataset créé: {len(dataset)} patients, vocab size: {dataset.vocab_size}")
    
    return dataset

# Fonction de test
def test_medical_dataset():
    """Test rapide du dataset médical"""
    
    print("🧪 Test du Medical Dataset")
    print("="*50)
    
    dataset = load_temporal_dataset('medical_sequences_pure.pkl', max_seq_length=50)
    
    if dataset is not None:
        # Créer un DataLoader pour tester
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"\n🔧 Test du DataLoader:")
        batch = next(iter(dataloader))
        
        print(f"   Batch size: {len(batch['patient_id'])}")
        print(f"   Input IDs shape: {batch['input_ids'].shape}")
        print(f"   Attention mask shape: {batch['attention_mask'].shape}")
        print(f"   Metadata shape: {batch['metadata'].shape}")
        
        # Vérifier que le padding fonctionne
        print(f"\n✅ Vérification padding:")
        print(f"   Input IDs[0]: {batch['input_ids'][0][:10]}...")
        print(f"   Mask[0]: {batch['attention_mask'][0][:10]}...")
        
        return dataset
    

# ==================== 4. FONCTION D'ENTRAÎNEMENT AVEC TEMPORALITÉ ====================
def train_real_temporal_model():
    """Entraîne le modèle avec de VRAIES données - VERSION CORRIGÉE"""
    
    print("=" * 60)
    print("🔥 ENTRAÎNEMENT RÉEL DU MODÈLE TEMPOREL")
    print("=" * 60)
    
    # 1. Charger les données
    dataset = load_temporal_dataset('medical_sequences_pure.pkl', max_seq_length=50)
    
    # Créer modèle
    model = MedicalTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        max_seq_length=50
    )
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"   - Train: {len(train_dataset)} patients")
    print(f"   - Validation: {len(val_dataset)} patients")
    print(f"   - Vocab codes: {dataset.code_encoder.classes_.shape[0]}")
    
    # 2. Créer le modèle
    print("\n2. 🧠 Création du modèle...")
    
    true_code_vocab_size = dataset.code_encoder.classes_.shape[0] + 1
    true_temporal_vocab_size = dataset.temporal_encoder.classes_.shape[0] + 1
    
    print(f"   - True code vocab size: {true_code_vocab_size}")
    print(f"   - True temporal vocab size: {true_temporal_vocab_size}")
    
    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Paramètres totaux: {total_params:,}")
    print(f"   - Paramètres entraînables: {trainable_params:,}")
    
    # 3. Configurer l'entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - Device: {device}")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Boucle d'entraînement - CORRECTION ICI
    print("\n3. 🔥 Début de l'entraînement...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        for batch_idx, batch in enumerate(train_loader):
            # Préparer les données
            input_ids = batch['input_ids'].to(device)
            temporal_ids = batch['temporal_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            
            batch_size, seq_len = input_ids.shape
            
            # CORRECTION: Masquer 15% des tokens MAIS calculer correctement
            mask_prob = 0.15
            mask = torch.rand(input_ids.shape, device=device) < mask_prob
            mask = mask & (input_ids != 0)  # Ne pas masquer le padding
            
            # Input masqué
            masked_input = input_ids.clone()
            masked_input[mask] = 0
            
            # Forward
            patient_emb = model(masked_input, attention_mask, temporal_ids, metadata)
            
            # CORRECTION: Prédire pour CHAQUE position, pas seulement une
            # Étendre patient_emb pour toutes les positions de séquence
            patient_emb_expanded = patient_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [16, 30, 96]
            
            # Calculer logits pour chaque position
            logits_all_positions = torch.matmul(patient_emb_expanded, model.code_embedding.weight.T)  # [16, 30, vocab_size]
            
            # CORRECTION: Sélectionner seulement les tokens masqués
            # Aplatir pour l'indexation
            logits_flat = logits_all_positions.reshape(-1, logits_all_positions.size(-1))  # [16*30, vocab_size]
            targets_flat = input_ids.reshape(-1)  # [16*30]
            mask_flat = mask.reshape(-1)  # [16*30]
            
            # Sélectionner seulement les positions masquées
            masked_logits = logits_flat[mask_flat]  # [n_masked, vocab_size]
            masked_targets = targets_flat[mask_flat]  # [n_masked]
            
            # Loss seulement sur tokens masqués
            if len(masked_targets) > 0:
                loss = criterion(masked_logits, masked_targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Afficher progression
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / max(batch_count, 1)
                n_masked = mask.sum().item() if mask.sum() > 0 else 0
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Masked: {n_masked}")
        
        # Loss moyenne epoch
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        print(f"✅ Training Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                temporal_ids = batch['temporal_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                metadata = batch['metadata'].to(device)
                
                batch_size, seq_len = input_ids.shape
                
                # Pas de masking en validation
                patient_emb = model(input_ids, attention_mask, temporal_ids, metadata)
                
                # CORRECTION: Même logique pour la validation
                patient_emb_expanded = patient_emb.unsqueeze(1).expand(-1, seq_len, -1)
                logits = torch.matmul(patient_emb_expanded, model.code_embedding.weight.T)
                
                # Loss sur tous tokens (pas seulement masqués)
                loss = criterion(
                    logits.view(-1, logits.size(-1)), 
                    input_ids.view(-1)
                )
                
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / max(val_count, 1)
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")
        
        # Sauvegarde checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'code_vocab_size': model.code_embedding.num_embeddings,
                'temporal_vocab_size': model.temporal_embedding.num_embeddings if hasattr(model, 'temporal_embedding') else 0,
                'code_encoder': dataset.code_encoder,
                'temporal_encoder': dataset.temporal_encoder,
                'config': {
                    'embed_dim': 96,
                    'num_heads': 4,
                    'num_layers': 2,
                    'max_seq_length': 30
                }
            }
            
            torch.save(checkpoint, f'temporal_model_checkpoint_epoch_{epoch+1}.pth')
            print(f"💾 Checkpoint sauvegardé: epoch_{epoch+1}")
    
    # 5. Sauvegarde finale
    print("\n4. 💾 Sauvegarde du modèle final...")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'code_encoder': dataset.code_encoder,
        'temporal_encoder': dataset.temporal_encoder,
        'config': {
            'code_vocab_size': model.code_embedding.num_embeddings,
            'temporal_vocab_size': model.temporal_embedding.num_embeddings if hasattr(model, 'temporal_embedding') else 0,
            'embed_dim': 96,
            'num_heads': 4,
            'num_layers': 2,
            'max_seq_length': 30
        }
    }
    
    torch.save(final_checkpoint, 'temporal_transformer_final.pth')
    print("✅ Modèle final sauvegardé!")
    
    return model, dataset

def train_medical_transformer():
    """Entraîne le modèle avec uniquement des codes médicaux - VERSION CORRIGÉE"""
    
    print("=" * 60)
    print("🔥 ENTRAÎNEMENT DU MODÈLE MÉDICAL PUR")
    print("=" * 60)
    
    # 1. Charger les données
    dataset = load_temporal_dataset('medical_sequences_pure.pkl', max_seq_length=50)
    
    if dataset is None:
        print("❌ Erreur de chargement des données")
        return
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"📊 Données:")
    print(f"   - Train: {len(train_dataset)} patients")
    print(f"   - Validation: {len(val_dataset)} patients")
    print(f"   - Vocabulaire médical: {dataset.vocab_size} codes")
    
    # 2. Créer le modèle
    print("\n🧠 Création du modèle MedicalTransformer...")
    
    # Utiliser vocab_size du dataset (déjà calculé avec padding)
    model = MedicalTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        max_seq_length=50
    )
    
    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Paramètres totaux: {total_params:,}")
    print(f"   - Paramètres entraînables: {trainable_params:,}")
    
    # 3. Configurer l'entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - Device: {device}")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Boucle d'entraînement
    print("\n🔥 Début de l'entraînement...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        for batch_idx, batch in enumerate(train_loader):
            # RÉCUPÉRATION DES DONNÉES CORRECTE
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Metadata optionnelle
            metadata = batch.get('metadata')
            if metadata is not None:
                metadata = metadata.to(device)
            
            batch_size, seq_len = input_ids.shape
            
            # Masked language modeling (masquer 15% des tokens)
            mask_prob = 0.15
            mask = torch.rand(input_ids.shape, device=device) < mask_prob
            mask = mask & (input_ids != 0)  # Ne pas masquer le padding
            
            # Input masqué
            masked_input = input_ids.clone()
            masked_input[mask] = 0
            
            # Forward (MODÈLE SIMPLIFIÉ)
            patient_emb = model(masked_input, attention_mask)
            
            # Prédire les tokens masqués
            # Étendre l'embedding patient à chaque position
            patient_emb_expanded = patient_emb.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Calculer logits (similarité avec embeddings de vocabulaire)
            logits = torch.matmul(patient_emb_expanded, model.code_embedding.weight.T)
            
            # Aplatir pour calcul de loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = input_ids.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            # Sélectionner seulement les tokens masqués
            masked_logits = logits_flat[mask_flat]
            masked_targets = targets_flat[mask_flat]
            
            # Loss seulement sur tokens masqués
            if len(masked_targets) > 0:
                loss = criterion(masked_logits, masked_targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Afficher progression
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / max(batch_count, 1)
                n_masked = mask.sum().item()
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Masqués: {n_masked}")
        
        # Loss moyenne epoch
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        print(f"✅ Training Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                metadata = batch.get('metadata')
                if metadata is not None:
                    metadata = metadata.to(device)
                
                batch_size, seq_len = input_ids.shape
                
                # Pas de masking en validation
                patient_emb = model(input_ids, attention_mask)
                
                # Calculer logits pour toutes positions
                patient_emb_expanded = patient_emb.unsqueeze(1).expand(-1, seq_len, -1)
                logits = torch.matmul(patient_emb_expanded, model.code_embedding.weight.T)
                
                # Loss sur tous tokens (pas seulement masqués)
                loss = criterion(
                    logits.view(-1, logits.size(-1)), 
                    input_ids.view(-1)
                )
                
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / max(val_count, 1)
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")
        
        # Sauvegarde checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'vocab_size': model.code_embedding.num_embeddings,
                'code_encoder': dataset.code_encoder,
                'config': {
                    'embed_dim': 128,
                    'num_heads': 4,
                    'num_layers': 3,
                    'max_seq_length': 50
                }
            }
            
            torch.save(checkpoint, f'medical_model_checkpoint_epoch_{epoch+1}.pth')
            print(f"💾 Checkpoint sauvegardé: epoch_{epoch+1}")
    
    # 5. Sauvegarde finale
    print("\n💾 Sauvegarde du modèle final...")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': model.code_embedding.num_embeddings,
        'code_encoder': dataset.code_encoder,
        'config': {
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'max_seq_length': 50
        }
    }
    
    torch.save(final_checkpoint, 'medical_transformer_final.pth')
    print("✅ Modèle final sauvegardé!")

    return model, dataset

def train_medical_model_safe():
    """Version ultra-simplifiée et sécurisée avec vraie loss"""
    
    print("🔥 ENTRAÎNEMENT SÉCURISÉ AVEC VRAIE LOSS")
    print("="*50)
    
    # 1. Créer un dataset simplifié
    class SimpleMedicalDataset(torch.utils.data.Dataset):
        def __init__(self, data_path, max_len=30):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.sequences = data['sequences']
            self.patient_info = data.get('patient_info', {})
            self.patient_ids = list(self.sequences.keys())
            self.max_len = max_len
            
            # Vocabulaire
            all_codes = []
            for seq in self.sequences.values():
                all_codes.extend(seq[:max_len])
            
            # Ajouter [PAD] et [MASK] tokens
            all_codes = ['[PAD]', '[MASK]'] + all_codes
            
            self.encoder = LabelEncoder()
            self.encoder.fit(all_codes)
            self.vocab_size = len(self.encoder.classes_)
            
            print(f"Dataset créé:")
            print(f"  - Patients: {len(self.patient_ids)}")
            print(f"  - Vocab size: {self.vocab_size}")
            print(f"  - Max sequence length: {max_len}")
        
        def __len__(self):
            return len(self.patient_ids)
        
        def __getitem__(self, idx):
            pid = self.patient_ids[idx]
            seq = self.sequences[pid]
            
            # Tronquer si trop long
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            
            # Encoder (+2 car [PAD]=0, [MASK]=1)
            encoded = self.encoder.transform(seq) + 2
            
            # Padding fixe
            if len(encoded) < self.max_len:
                encoded = np.pad(encoded, (0, self.max_len - len(encoded)))
            
            # Convertir en tensor
            input_ids = torch.tensor(encoded, dtype=torch.long)
            
            # Masque d'attention
            real_len = min(len(seq), self.max_len)
            attention_mask = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask[:real_len] = 1
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'patient_id': pid
            }
    
    # 2. Charger dataset
    print("\n📂 Chargement des données...")
    dataset = SimpleMedicalDataset('medical_sequences_pure.pkl', max_len=30)
    
    # 3. Créer modèle
    print("\n🧠 Création du modèle...")
    model = MedicalTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=64,  # Petit pour test rapide
        num_heads=2,
        num_layers=2,
        max_seq_length=30
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model.to(device)
    
    # 4. DataLoader avec collate_fn
    def simple_collate(batch):
        """Fonction de collate personnalisée"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        patient_ids = [item['patient_id'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'patient_id': patient_ids
        }
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=simple_collate,
        num_workers=0  # Éviter les problèmes de multiprocessing
    )
    
    # 5. Configurer l'optimizer et la loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer [PAD] token (index 0)
    
    print("\n🔥 Début de l'entraînement...")
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        total_masked = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Transférer sur device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_size, seq_len = input_ids.shape
            
            # 1. Créer des targets (copie de input_ids)
            targets = input_ids.clone()
            
            # 2. Créer un masque aléatoire (15% des tokens non-padding)
            mask_prob = 0.15
            mask = torch.rand(input_ids.shape, device=device) < mask_prob
            mask = mask & (input_ids != 0)  # Ne pas masquer les tokens [PAD]
            
            # 3. Remplacer les tokens masqués par [MASK] token (index 1)
            masked_input = input_ids.clone()
            masked_input[mask] = 1  # [MASK] token
            
            # 4. Forward pass
            patient_emb = model(masked_input, attention_mask)
            
            # 5. Projeter pour obtenir les logits
            # Étendre l'embedding patient à chaque position
            patient_emb_expanded = patient_emb.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Calculer les logits (similarité avec embeddings de vocabulaire)
            logits = torch.matmul(patient_emb_expanded, model.code_embedding.weight.T)
            
            # 6. Calculer la loss SEULEMENT sur les tokens masqués
            # Aplatir les tensors
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            # Sélectionner seulement les positions masquées
            if mask_flat.sum() > 0:  # S'il y a des tokens masqués
                masked_logits = logits_flat[mask_flat]
                masked_targets = targets_flat[mask_flat]
                
                # Calculer la loss
                loss = criterion(masked_logits, masked_targets)
                
                # 7. Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_masked += mask_flat.sum().item()
            
            # Afficher progression
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        # Afficher résultats de l'epoch
        avg_loss = total_loss / len(dataloader)
        avg_masked = total_masked / len(dataloader) if len(dataloader) > 0 else 0
        print(f"\n✅ Epoch {epoch+1}/5:")
        print(f"   Loss moyenne: {avg_loss:.4f}")
        print(f"   Tokens masqués moyen par batch: {avg_masked:.1f}")
    
    # 6. Test du modèle
    print("\n🧪 Test du modèle...")
    model.eval()
    
    with torch.no_grad():
        # Prendre un batch de test
        test_batch = next(iter(dataloader))
        input_ids = test_batch['input_ids'].to(device)
        attention_mask = test_batch['attention_mask'].to(device)
        
        # Générer embeddings
        embeddings = model(input_ids, attention_mask)
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Exemple embedding (premier patient): {embeddings[0][:5]}...")
    
    # 7. Sauvegarder le modèle
    print("\n💾 Sauvegarde du modèle...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'embed_dim': 64,
        'encoder': dataset.encoder,
        'config': {
            'max_seq_length': 30,
            'num_heads': 2,
            'num_layers': 2
        }
    }, 'medical_model_safe.pth')
    
    print("✅ Entraînement terminé!")
    print("📁 Modèle sauvegardé: medical_model_safe.pth")
    
    return model, dataset

# Version encore PLUS simple pour déboguer
def train_minimal():
    """Version minimale pour déboguer"""
    
    print("🧪 VERSION MINIMALE POUR DÉBOGUER")
    
    # 1. Créer des données factices pour tester
    vocab_size = 100
    batch_size = 4
    seq_len = 20
    
    # Données factices
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Données factices:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")
    
    # 2. Modèle minimal
    model = MedicalTransformer(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        max_seq_length=seq_len
    )
    
    device = torch.device('cpu')
    model.to(device)
    
    # 3. Test forward pass
    print("\n🧠 Test forward pass...")
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  ✅ Forward pass réussi!")
    
    # 4. Test avec backward pass
    print("\n🔙 Test backward pass...")
    model.train()
    
    # Créer une loss simple
    dummy_target = torch.randn(embeddings.shape)
    loss_fn = nn.MSELoss()
    
    # Forward
    embeddings = model(input_ids, attention_mask)
    loss = loss_fn(embeddings, dummy_target)
    
    # Backward
    loss.backward()
    print(f"  ✅ Backward pass réussi!")
    print(f"  Loss: {loss.item():.4f}")
    
    # Vérifier les gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  Gradient {name}: {param.grad.abs().mean():.6f}")
    
    return model

# 6. Génération des embeddings pour clustering
def generate_embeddings(model, dataset, device='cpu'):
    """S'adapte automatiquement au format du dataset"""
    
    print("\n🧬 Génération des embeddings (mode adaptatif)...")
    
    # Analyser le dataset
    sample = dataset[0]
    has_temporal = 'temporal_ids' in sample
    has_metadata = 'metadata' in sample
    
    print(f"📊 Format détecté:")
    print(f"   - Temporal ids: {'OUI' if has_temporal else 'NON'}")
    print(f"   - Metadata: {'OUI' if has_metadata else 'NON'}")
    
    model.eval()
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    embeddings = {}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            patient_ids = batch['patient_id']
            
            # Préparer les arguments selon ce qui existe
            kwargs = {}
            
            if has_temporal:
                kwargs['temporal_ids'] = batch['temporal_ids'].to(device)
            
            if has_metadata:
                kwargs['metadata'] = batch['metadata'].to(device)
            
            # Appel dynamique
            patient_emb = model(input_ids, attention_mask, **kwargs)
            
            for i, pid in enumerate(patient_ids):
                embeddings[pid] = patient_emb[i].cpu().numpy()
    
    print(f"✅ {len(embeddings)} embeddings générés")
    return embeddings

def generate_medical_embeddings_simple(model, dataset, device='cpu'):
    """Version simplifiée pour modèle MedicalTransformer"""
    
    print("\n🧬 Génération des embeddings médicaux...")
    
    if isinstance(model, tuple):
        print("⚠️  Extraction du modèle depuis tuple...")
        model = model[0]
    
    model.eval()
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    embeddings = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            patient_ids = batch['patient_id']
            
            # Simple appel pour MedicalTransformer
            patient_emb = model(input_ids, attention_mask)
            
            for i, pid in enumerate(patient_ids):
                embeddings[pid] = patient_emb[i].cpu().numpy()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(dataloader)}")
    
    print(f"✅ {len(embeddings)} embeddings générés")
    
    # Sauvegarder
    with open('medical_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def diagnose_dataset():
    """Diagnostique les problèmes de dataset"""
    
    print("🔍 DIAGNOSTIC DU DATASET")
    print("="*50)
    
    # Test load_temporal_dataset
    dataset = load_temporal_dataset('medical_sequences_pure.pkl', max_seq_length=50)
    
    if dataset:
        # Vérifier 10 échantillons
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            print(f"\nÉchantillon {i}:")
            print(f"  input_ids shape: {item['input_ids'].shape}")
            print(f"  attention_mask shape: {item['attention_mask'].shape}")
            
            # Vérifier que les séquences ont la longueur attendue
            if item['input_ids'].shape[0] != 50:
                print(f"  ⚠️  Longueur inattendue: {item['input_ids'].shape[0]} au lieu de 50")
            
            # Vérifier le padding
            mask_sum = item['attention_mask'].sum().item()
            print(f"  Tokens réels: {mask_sum}/{50}")
            
            if 'metadata' in item:
                print(f"  metadata shape: {item['metadata'].shape}")

# Script principal
if __name__ == "__main__":
    print("Options:")
    print("1. Entraînement sécurisé complet")
    print("2. Version minimale de débogage")
    print("3. Tester seulement le forward/backward")
    
    choice = input("Choix (1-3): ")
    
    if choice == "1":
        model, dataset = train_medical_model_safe()
        
        # Générer des embeddings pour clustering
        print("\n🧬 Génération des embeddings pour clustering...")
        
        # Charger tout le dataset dans un DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False,
            collate_fn=lambda b: {
                'input_ids': torch.stack([item['input_ids'] for item in b]),
                'attention_mask': torch.stack([item['attention_mask'] for item in b]),
                'patient_id': [item['patient_id'] for item in b]
            }
        )
        
        device = next(model.parameters()).device
        model.eval()
        
        embeddings_dict = {}
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                patient_ids = batch['patient_id']
                
                patient_emb = model(input_ids, attention_mask)
                
                for i, pid in enumerate(patient_ids):
                    embeddings_dict[pid] = patient_emb[i].cpu().numpy()
        
        # Sauvegarder les embeddings
        with open('medical_embeddings_safe.pkl', 'wb') as f:
            pickle.dump({
                'embeddings': embeddings_dict,
                'patient_ids': list(embeddings_dict.keys())
            }, f)
        
        print(f"✅ {len(embeddings_dict)} embeddings générés!")
        print("📁 Fichier: medical_embeddings_safe.pkl")
        
    elif choice == "2":
        train_minimal()
    elif choice == "3":
        # Juste tester forward/backward
        vocab_size = 50
        model = MedicalTransformer(
            vocab_size=vocab_size,
            embed_dim=16,
            num_heads=1,
            num_layers=1,
            max_seq_length=10
        )
        
        # Test simple
        input_ids = torch.randint(1, vocab_size, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        print("Test forward...")
        output = model(input_ids, attention_mask)
        print(f"Output shape: {output.shape}")
        
        print("\nTest backward...")
        loss = output.sum()
        loss.backward()
        print("✅ Backward réussi!")
    