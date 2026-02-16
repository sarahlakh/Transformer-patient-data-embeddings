# transformer_paper_methodology.py

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')
import time
import json
from collections import defaultdict

# ==================== TRANSFORMER COMME DANS LE PAPIER ====================

class BEHRTLikeTransformer(nn.Module):
    """Transformer inspiré de BEHRT (mentionné dans le papier)"""
    def __init__(self, vocab_size, embed_dim=128, n_heads=8, n_layers=4, max_seq_len=100):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings comme dans BEHRT
        self.code_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Segment embedding (pour différencier diagnostic/traitement)
        self.segment_embedding = nn.Embedding(3, embed_dim)  # 0=diagnostic, 1=traitement, 2=autre
        
        # Transformer Encoder (comme dans l'article)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Pooling pour représentation patient (comme dans Deep Patient)
        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        
        # Pour la tâche de prédiction (prétraining)
        self.code_predictor = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, segment_ids=None, attention_mask=None, return_all=False):
        batch_size, seq_len = input_ids.shape
        
        # Code embeddings
        code_emb = self.code_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Segment embeddings
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        seg_emb = self.segment_embedding(segment_ids)
        
        # Somme des embeddings
        embeddings = code_emb + pos_emb + seg_emb
        
        # Attention mask
        if attention_mask is None:
            attention_mask = (input_ids != 0)
        
        # Transformer
        transformer_output = self.transformer(
            embeddings,
            src_key_padding_mask=~attention_mask
        )
        
        # Pooling: utiliser le premier token (comme [CLS] dans BERT)
        pooled_output = self.pooler(transformer_output[:, 0, :])
        
        # Prédiction pour prétraining
        prediction_logits = self.code_predictor(pooled_output)
        
        if return_all:
            return {
                'last_hidden_state': transformer_output,
                'pooled_output': pooled_output,
                'prediction_logits': prediction_logits
            }
        else:
            return pooled_output

# ==================== PRÉPARATION DES DONNÉES (COMme Section III) ====================

def prepare_paper_dataset(sequences, patient_info=None, max_visits=50, max_codes_per_visit=10):
    """Prépare le dataset exactement comme dans le papier"""
    print("="*60)
    print("PRÉPARATION DES DONNÉES (Section III du papier)")
    print("="*60)
    
    # 1. Filtrer les patients (comme dans Section III)
    print("\n1. Filtrage des patients...")
    filtered_sequences = {}
    filtered_info = {}
    
    for pid, seq in sequences.items():
        # Garder seulement les patients avec au moins 4 visites (comme min dans papier)
        if len(seq) >= 4:
            filtered_sequences[pid] = seq
            if patient_info and pid in patient_info:
                filtered_info[pid] = patient_info[pid]
    
    print(f"   Patients retenus: {len(filtered_sequences)}/{len(sequences)}")
    
    # 2. Créer le vocabulaire des codes médicaux
    print("\n2. Création du vocabulaire...")
    all_codes = []
    for seq in filtered_sequences.values():
        for visit in seq:
            if visit != 'NO_CODE':
                all_codes.append(visit)
    
    code_encoder = LabelEncoder()
    code_encoder.fit(all_codes)
    vocab_size = len(code_encoder.classes_)
    
    print(f"   Taille vocabulaire: {vocab_size} codes")
    print(f"   Codes uniques: {len(set(all_codes))}")
    
    # 3. Analyser les statistiques (comme Tableau dans Section III)
    print("\n3. Statistiques des données:")
    seq_lengths = [len(seq) for seq in filtered_sequences.values()]
    print(f"   Visites par patient: moyen={np.mean(seq_lengths):.1f}, "
          f"min={np.min(seq_lengths)}, max={np.max(seq_lengths)}")
    
    # 4. Préparer les séquences pour le Transformer
    print("\n4. Encodage des séquences...")
    encoded_sequences = []
    patient_ids_list = []
    segment_ids_list = []
    
    code_to_idx = {code: idx+1 for idx, code in enumerate(code_encoder.classes_)}  # +1 pour padding=0
    
    for pid, seq in filtered_sequences.items():
        # Limiter aux max_visits premières visites
        seq = seq[:max_visits]
        
        encoded_seq = []
        segment_seq = []
        
        for visit_idx, visit in enumerate(seq):
            if visit != 'NO_CODE' and visit in code_to_idx:
                # Encoder le code
                code_idx = code_to_idx[visit]
                encoded_seq.append(code_idx)
                
                # Segment ID (simplifié: tout à 0 pour commencer)
                segment_seq.append(0)
        
        if len(encoded_seq) > 0:
            # Padding à max_seq_len
            if len(encoded_seq) > max_codes_per_visit * max_visits:
                encoded_seq = encoded_seq[:max_codes_per_visit * max_visits]
                segment_seq = segment_seq[:max_codes_per_visit * max_visits]
            
            encoded_sequences.append(encoded_seq)
            segment_ids_list.append(segment_seq)
            patient_ids_list.append(pid)
    
    # Padding final
    max_seq_len = max_codes_per_visit * max_visits
    padded_sequences = np.zeros((len(encoded_sequences), max_seq_len), dtype=np.int32)
    padded_segments = np.zeros((len(encoded_sequences), max_seq_len), dtype=np.int32)
    
    for i, (seq, seg) in enumerate(zip(encoded_sequences, segment_ids_list)):
        if len(seq) > 0:
            padded_sequences[i, :len(seq)] = seq[:max_seq_len]
            padded_segments[i, :len(seg)] = seg[:max_seq_len]
    
    print(f"   Séquences préparées: {len(padded_sequences)}")
    print(f"   Longueur max séquence: {max_seq_len}")
    
    return {
        'input_ids': torch.tensor(padded_sequences, dtype=torch.long),
        'segment_ids': torch.tensor(padded_segments, dtype=torch.long),
        'patient_ids': patient_ids_list,
        'code_encoder': code_encoder,
        'vocab_size': vocab_size + 1,  # +1 pour padding
        'max_seq_len': max_seq_len,
        'patient_info': filtered_info
    }

# ==================== ENTRAÎNEMENT (COMme Section IV) ====================

def train_transformer_paper_method(data, epochs=10, batch_size=32, embed_dim=128):
    """Entraîne le Transformer comme dans le papier"""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TRANSFORMER (Section IV)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Créer le modèle
    model = BEHRTLikeTransformer(
        vocab_size=data['vocab_size'],
        embed_dim=embed_dim,
        n_heads=8,
        n_layers=4,
        max_seq_len=data['max_seq_len']
    ).to(device)
    
    # Split train/validation (80/20 comme dans papier)
    print("\nSplit train/validation (80/20)...")
    n_samples = len(data['input_ids'])
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_data = {
        'input_ids': data['input_ids'][train_idx].to(device),
        'segment_ids': data['segment_ids'][train_idx].to(device),
        'patient_ids': [data['patient_ids'][i] for i in train_idx]
    }
    
    val_data = {
        'input_ids': data['input_ids'][val_idx].to(device),
        'segment_ids': data['segment_ids'][val_idx].to(device),
        'patient_ids': [data['patient_ids'][i] for i in val_idx]
    }
    
    print(f"   Train: {len(train_idx)} patients")
    print(f"   Validation: {len(val_idx)} patients")
    
    # Optimizer et scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer padding
    
    # Entraînement
    print("\nDébut de l'entraînement...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Mode train
        model.train()
        train_loss = 0
        train_batches = 0
        
        # Mélanger les données d'entraînement
        train_indices = torch.randperm(len(train_idx))
        
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_indices[i:i+batch_size]
            
            batch_input = train_data['input_ids'][batch_idx]
            batch_segment = train_data['segment_ids'][batch_idx]
            batch_attention = (batch_input != 0)
            
            # Forward pass
            outputs = model(batch_input, batch_segment, batch_attention, return_all=True)
            predictions = outputs['prediction_logits']
            
            # Target: prédire les codes du patient
            targets = batch_input.flatten()
            predictions = predictions.repeat_interleave(batch_input.shape[1], dim=0)
            
            # Loss
            loss = criterion(predictions, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_input = val_data['input_ids'][i:i+batch_size]
                batch_segment = val_data['segment_ids'][i:i+batch_size]
                batch_attention = (batch_input != 0)
                
                outputs = model(batch_input, batch_segment, batch_attention, return_all=True)
                predictions = outputs['prediction_logits']
                
                targets = batch_input.flatten()
                predictions = predictions.repeat_interleave(batch_input.shape[1], dim=0)
                
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_batches += 1
        
        # Métriques epoch
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n✓ Entraînement terminé en {total_time:.1f} secondes")
    
    return model, {'train_idx': train_idx, 'val_idx': val_idx}

# ==================== GÉNÉRATION DES EMBEDDINGS ====================

def generate_patient_embeddings(model, data, split_indices=None):
    """Génère les embeddings patients comme dans le papier"""
    print("\n" + "="*60)
    print("GÉNÉRATION DES EMBEDDINGS PATIENTS")
    print("="*60)
    
    device = next(model.parameters()).device
    model.eval()
    
    all_embeddings = {}
    
    with torch.no_grad():
        batch_size = 64
        
        for i in range(0, len(data['input_ids']), batch_size):
            batch_input = data['input_ids'][i:i+batch_size].to(device)
            batch_segment = data['segment_ids'][i:i+batch_size].to(device)
            batch_attention = (batch_input != 0)
            batch_ids = data['patient_ids'][i:i+batch_size]
            
            # Générer embeddings
            embeddings = model(batch_input, batch_segment, batch_attention)
            
            for j, pid in enumerate(batch_ids):
                all_embeddings[pid] = embeddings[j].cpu().numpy()
            
            # Progress
            if (i // batch_size) % 10 == 0:
                print(f"  Traité {i}/{len(data['input_ids'])} patients", end='\r')
    
    print(f"\n✓ Embeddings générés: {len(all_embeddings)} patients")
    
    # Séparer train/val si demandé
    if split_indices:
        train_embeddings = {}
        val_embeddings = {}
        
        for i, pid in enumerate(data['patient_ids']):
            if i in split_indices['train_idx']:
                train_embeddings[pid] = all_embeddings[pid]
            else:
                val_embeddings[pid] = all_embeddings[pid]
        
        print(f"   Train: {len(train_embeddings)} patients")
        print(f"   Validation: {len(val_embeddings)} patients")
        
        return all_embeddings, train_embeddings, val_embeddings
    
    return all_embeddings

# ==================== CLUSTERING ET ÉVALUATION (COMme Section V) ====================

def evaluate_clustering_paper_method(embeddings_dict, n_clusters=5, method='kmeans', n_folds=10):
    """Évalue le clustering exactement comme dans le papier (Section V)"""
    print(f"\n" + "="*60)
    print(f"ÉVALUATION CLUSTERING (Section V) - {method.upper()}")
    print("="*60)
    
    # Convertir en array
    patient_ids = list(embeddings_dict.keys())
    X = np.array([embeddings_dict[pid] for pid in patient_ids])
    
    print(f"Données: {X.shape[0]} patients, {X.shape[1]} dimensions")
    
    # Normaliser comme dans le papier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {
        'silhouette_scores': [],
        'davies_bouldin_scores': [],
        'all_labels': []
    }
    
    # Validation croisée 10-fold comme dans le papier
    print(f"\nValidation croisée {n_folds}-fold...")
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        
        # Clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:  # GMM
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        
        # Entraîner sur train
        clusterer.fit(X_train)
        
        # Prédire sur test
        if method == 'kmeans':
            test_labels = clusterer.predict(X_test)
        else:
            test_labels = clusterer.predict(X_test)
        
        # Métriques sur test
        silhouette = silhouette_score(X_test, test_labels)
        davies = davies_bouldin_score(X_test, test_labels)
        
        results['silhouette_scores'].append(silhouette)
        results['davies_bouldin_scores'].append(davies)
        
        print(f"  Fold {fold}: Silhouette={silhouette:.3f}, Davies-Bouldin={davies:.3f}")
    
    # Métriques finales
    mean_silhouette = np.mean(results['silhouette_scores'])
    std_silhouette = np.std(results['silhouette_scores'])
    mean_davies = np.mean(results['davies_bouldin_scores'])
    std_davies = np.std(results['davies_bouldin_scores'])
    
    print(f"\n📊 RÉSULTATS {method.upper()} (moyenne ± std sur {n_folds} folds):")
    print(f"   Silhouette Score: {mean_silhouette:.3f} ± {std_silhouette:.3f}")
    print(f"   Davies-Bouldin Index: {mean_davies:.3f} ± {std_davies:.3f}")
    
    # Clustering final sur toutes les données
    if method == 'kmeans':
        final_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        final_clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
    
    final_labels = final_clusterer.fit_predict(X_scaled)
    
    # Distribution des clusters
    unique, counts = np.unique(final_labels, return_counts=True)
    print(f"\n📈 DISTRIBUTION DES CLUSTERS:")
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(final_labels) * 100
        print(f"   Cluster {cluster_id}: {count} patients ({percentage:.1f}%)")
    
    return {
        'mean_silhouette': mean_silhouette,
        'std_silhouette': std_silhouette,
        'mean_davies': mean_davies,
        'std_davies': std_davies,
        'final_labels': final_labels,
        'patient_ids': patient_ids,
        'embeddings': X,
        'clusterer': final_clusterer,
        'scaler': scaler
    }

# ==================== TEST CHI² (COMme Section V-C) ====================

def chi_squared_test_paper(clustering_results, patient_info, clinical_variables):
    """Test chi² exactement comme dans le papier (Section V-C)"""
    print("\n" + "="*60)
    print("TEST CHI-CARRÉ (Section V-C du papier)")
    print("="*60)
    
    labels = clustering_results['final_labels']
    patient_ids = clustering_results['patient_ids']
    n_clusters = len(np.unique(labels))
    
    print(f"Nombre de clusters: {n_clusters}")
    print(f"Variables cliniques testées: {len(clinical_variables)}")
    
    results = {}
    
    for var_name in clinical_variables:
        # Préparer tableau de contingence
        contingency_table = np.zeros((n_clusters, 2))  # 2 catégories simplifiées
        
        for i, pid in enumerate(patient_ids):
            if pid in patient_info and var_name in patient_info[pid]:
                value = patient_info[pid][var_name]
                cluster = labels[i]
                
                # Simplifier en binaire (0/1) pour le test
                if isinstance(value, (int, float)):
                    binary_value = 1 if value > np.median(list(patient_info[pid].values() 
                                                              for pid in patient_info 
                                                              if var_name in patient_info[pid])) else 0
                else:
                    # Pour les variables catégorielles
                    binary_value = 1 if str(value) != '0' and str(value).lower() != 'false' else 0
                
                contingency_table[cluster, binary_value] += 1
        
        # Test chi²
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            results[var_name] = {
                'chi2': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'expected_frequencies': expected.tolist()
            }
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{var_name:25s} p-value = {p_value:.6f} {significance}")
            
        except Exception as e:
            print(f"{var_name:25s} Erreur: {str(e)[:50]}...")
            results[var_name] = {
                'chi2': None,
                'p_value': None,
                'significant': False,
                'error': str(e)
            }
    
    # Compter les variables significatives
    significant_vars = [var for var, res in results.items() 
                       if res.get('significant', False)]
    
    print(f"\n📊 {len(significant_vars)}/{len(clinical_variables)} variables significatives (p < 0.05)")
    
    return results

# ==================== COMPARAISON AVEC MÉTHODES DU PAPIER ====================

def compare_with_paper_results(transformer_results, paper_results_path='benchmark_results.pkl'):
    """Compare les résultats avec les 3 méthodes du papier"""
    print("\n" + "="*80)
    print("COMPARAISON AVEC LES 3 MÉTHODES DU PAPIER")
    print("="*80)
    
    # Charger les résultats du papier
    try:
        with open(paper_results_path, 'rb') as f:
            paper_results = pickle.load(f)
        
        print("✅ Résultats des 3 méthodes chargés")
        
        # Extraire les scores
        paper_scores = {}
        for method in ['SkipGram', 'Med2Vec', 'DeepPatient']:
            if method in paper_results:
                paper_scores[method] = paper_results[method]['kmeans']['metrics']['silhouette']
        
    except FileNotFoundError:
        print("⚠️  Fichier des résultats papier non trouvé, utilisation des scores rapportés")
        # Scores du papier (Table III)
        paper_scores = {
            'SkipGram': 0.60,
            'Med2Vec': 0.55,
            'DeepPatient': 0.98
        }
    
    # Vos scores Transformer
    transformer_score = transformer_results['kmeans']['mean_silhouette']
    
    print(f"\n📊 TABLEAU COMPARATIF (Silhouette Score):")
    print("-"*60)
    print(f"{'Méthode':<20} {'Score':<15} {'Source':<10}")
    print("-"*60)
    
    for method, score in paper_scores.items():
        source = "Papier" if score in [0.60, 0.55, 0.98] else "Votre implémentation"
        print(f"{method:<20} {score:<15.3f} {source:<10}")
    
    print(f"{'Transformer':<20} {transformer_score:<15.3f} {'Votre étude':<10}")
    
    print("\n🔍 ANALYSE COMPARATIVE:")
    print("-"*60)
    
    # Comparaison avec votre SkipGram
    your_skipgram = paper_scores.get('SkipGram', 0.243)  # Votre résultat ou celui du papier
    improvement = transformer_score - your_skipgram
    improvement_pct = (improvement / your_skipgram * 100) if your_skipgram > 0 else 0
    
    if improvement > 0:
        print(f"✅ Transformer surpasse SkipGram de {improvement:.3f} points ({improvement_pct:.1f}%)")
    else:
        print(f"⚠️  SkipGram reste meilleur de {-improvement:.3f} points")
    
    # Comparaison avec DeepPatient du papier
    paper_deeppatient = 0.98
    diff_to_paper = transformer_score - paper_deeppatient
    
    print(f"\n📈 Comparaison avec les résultats du papier:")
    print(f"   Différence avec Deep Patient (papier): {diff_to_paper:+.3f}")
    
    if transformer_score > 0.7:
        print("   ✅ Performance comparable au papier")
    elif transformer_score > 0.5:
        print("   👍 Bonne performance, mais inférieure au papier")
    else:
        print("   ⚠️  Performance significativement inférieure au papier")
        print("      Raisons possibles: données différentes, paramètres, prétraitement")
    
    return {
        'paper_scores': paper_scores,
        'transformer_score': transformer_score,
        'improvement_vs_skipgram': improvement,
        'difference_vs_paper': diff_to_paper
    }

# ==================== RAPPORT FINAL (COMme Section VI) ====================

def generate_final_report(transformer_results, chi2_results, comparison_results):
    """Génère un rapport final comme dans le papier"""
    
    report = {
        'title': "Benchmark du Transformer pour l'Encodage des Parcours Patients",
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'methodology': "Basé sur l'article: 'Encoding breast cancer patients' medical pathways...'",
        'results': {
            'clustering_performance': {
                'kmeans': {
                    'silhouette_mean': transformer_results['kmeans']['mean_silhouette'],
                    'silhouette_std': transformer_results['kmeans']['std_silhouette'],
                    'davies_mean': transformer_results['kmeans']['mean_davies'],
                    'davies_std': transformer_results['kmeans']['std_davies']
                },
                'gmm': {
                    'silhouette_mean': transformer_results['gmm']['mean_silhouette'],
                    'silhouette_std': transformer_results['gmm']['std_silhouette'],
                    'davies_mean': transformer_results['gmm']['mean_davies'],
                    'davies_std': transformer_results['gmm']['std_davies']
                }
            },
            'clinical_significance': {
                'total_variables_tested': len(chi2_results),
                'significant_variables': sum(1 for r in chi2_results.values() if r.get('significant', False)),
                'significant_variables_list': [var for var, res in chi2_results.items() 
                                             if res.get('significant', False)]
            },
            'comparison': comparison_results
        },
        'conclusion': ""
    }
    
    # Conclusion basée sur les résultats
    silhouette = transformer_results['kmeans']['mean_silhouette']
    clinical_sig = report['results']['clinical_significance']['significant_variables']
    
    if silhouette > 0.5 and clinical_sig > 0:
        report['conclusion'] = "Le Transformer fournit à la fois un bon clustering et une signification clinique."
    elif silhouette > 0.5:
        report['conclusion'] = "Bon clustering mais signification clinique limitée (comme Deep Patient dans le papier)."
    elif clinical_sig > 0:
        report['conclusion'] = "Signification clinique bonne mais clustering faible (comme Med2Vec dans le papier)."
    else:
        report['conclusion'] = "Performances limitées sur les deux aspects."
    
    # Sauvegarder le rapport
    with open('transformer_paper_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("📄 RAPPORT FINAL GÉNÉRÉ")
    print("="*80)
    print(f"\nFichier: transformer_paper_report.json")
    print(f"\nCONCLUSION: {report['conclusion']}")
    
    return report

# ==================== FONCTION PRINCIPALE ====================

def main():
    """Fonction principale suivant exactement la méthodologie du papier"""
    
    print("="*80)
    print("BENCHMARK TRANSFORMER - MÉTHODOLOGIE DU PAPIER")
    print("="*80)
    
    total_start = time.time()
    
    # 1. Charger les données
    print("\n📂 ÉTAPE 1: Chargement des données")
    print("-"*40)
    try:
        with open('medical_sequences_pure.pkl', 'rb') as f:
            data = pickle.load(f)
        
        sequences = data['sequences']
        patient_info = data.get('patient_info', {})
        
        print(f"✅ Patients: {len(sequences)}")
        print(f"✅ Informations patients: {len(patient_info)}")
        
    except FileNotFoundError:
        print("❌ Fichier medical_sequences_pure.pkl non trouvé!")
        return
    
    # 2. Préparer le dataset (Section III)
    print("\n📊 ÉTAPE 2: Préparation du dataset (Section III)")
    print("-"*40)
    paper_data = prepare_paper_dataset(sequences, patient_info)
    
    # 3. Entraîner le Transformer (Section IV)
    print("\n🤖 ÉTAPE 3: Entraînement Transformer (Section IV)")
    print("-"*40)
    transformer_model, split_indices = train_transformer_paper_method(
        paper_data, 
        epochs=5,  # Réduit pour rapidité, augmenter pour meilleurs résultats
        batch_size=32,
        embed_dim=128
    )
    
    # 4. Générer les embeddings
    print("\n🧬 ÉTAPE 4: Génération des embeddings patients")
    print("-"*40)
    all_embeddings = generate_patient_embeddings(transformer_model, paper_data)
    
    # 5. Évaluer le clustering (Section V)
    print("\n🎯 ÉTAPE 5: Évaluation du clustering (Section V)")
    print("-"*40)
    
    # K-means
    kmeans_results = evaluate_clustering_paper_method(
        all_embeddings, 
        n_clusters=5, 
        method='kmeans',
        n_folds=5  # Réduit pour rapidité, mettre 10 comme dans le papier
    )
    
    # GMM
    gmm_results = evaluate_clustering_paper_method(
        all_embeddings,
        n_clusters=5,
        method='gmm',
        n_folds=5
    )
    
    transformer_results = {
        'kmeans': kmeans_results,
        'gmm': gmm_results
    }
    
    # 6. Test chi² (Section V-C)
    print("\n🔬 ÉTAPE 6: Test chi² (Section V-C)")
    print("-"*40)
    
    # Variables cliniques à tester
    clinical_vars = []
    if patient_info:
        sample_pid = next(iter(patient_info.keys()))
        clinical_vars = list(patient_info[sample_pid].keys())
        # Limiter pour la démonstration
        clinical_vars = clinical_vars[:10]
    
    if clinical_vars:
        chi2_results = chi_squared_test_paper(
            kmeans_results,
            paper_data['patient_info'],
            clinical_vars
        )
    else:
        print("⚠️  Aucune information patient disponible pour le test chi²")
        chi2_results = {}
    
    # 7. Comparaison avec le papier
    print("\n📈 ÉTAPE 7: Comparaison avec les résultats du papier")
    print("-"*40)
    comparison_results = compare_with_paper_results(transformer_results)
    
    # 8. Rapport final
    print("\n📄 ÉTAPE 8: Génération du rapport final")
    print("-"*40)
    report = generate_final_report(transformer_results, chi2_results, comparison_results)
    
    # 9. Sauvegarde complète
    print("\n💾 ÉTAPE 9: Sauvegarde des résultats complets")
    print("-"*40)
    
    final_results = {
        'transformer_results': transformer_results,
        'chi2_results': chi2_results,
        'comparison_results': comparison_results,
        'report': report,
        'config': {
            'embed_dim': 128,
            'n_layers': 4,
            'n_heads': 8,
            'epochs': 5,
            'batch_size': 32
        },
        'embeddings_sample': {k: v for k, v in list(all_embeddings.items())[:1000]}
    }
    
    with open('transformer_paper_methodology_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    total_time = time.time() - total_start
    print(f"\n" + "="*80)
    print(f"✅ ÉTUDE COMPLÉTÉE EN {total_time:.1f} SECONDES")
    print("="*80)
    
    print("\n📁 FICHIERS GÉNÉRÉS:")
    print("   - transformer_paper_report.json : Rapport complet")
    print("   - transformer_paper_methodology_results.pkl : Résultats détaillés")
    
    print("\n🎯 CONCLUSION FINALE DE L'ÉTUDE:")
    print("-"*40)
    
    silhouette = transformer_results['kmeans']['mean_silhouette']
    clinical_sig = report['results']['clinical_significance']['significant_variables']
    
    print(f"1. Score Silhouette: {silhouette:.3f}")
    print(f"2. Variables cliniques significatives: {clinical_sig}")
    
    if silhouette > 0.5 and clinical_sig > 0:
        print("3. ✅ TRANSFORMER RECOMMANDÉ: Bon clustering ET signification clinique")
    elif silhouette > 0.5:
        print("3. ⚠️  Bon clustering mais vérifier la signification clinique")
    else:
        print("3. ❌ Clustering faible - Explorer d'autres méthodes")

if __name__ == "__main__":
    main()