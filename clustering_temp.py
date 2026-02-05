import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from collections import Counter

def load_medical_embeddings(embeddings_path='medical_embeddings.pkl'):
    """Charge les embeddings médicaux"""
    
    print(f" Chargement des embeddings depuis {embeddings_path}...")   
    try:
        with open(embeddings_path, 'rb') as f:
            emb_data = pickle.load(f)
        
        print(f"v Fichier chargé, type: {type(emb_data)}")
        
        # Gérer différents formats
        if isinstance(emb_data, dict):
            # Format 1: dict avec clés 'embeddings' et 'patient_ids'
            if 'embeddings' in emb_data:
                embeddings_dict = emb_data['embeddings']
                if 'patient_ids' in emb_data:
                    patient_ids = emb_data['patient_ids']
                else:
                    patient_ids = list(embeddings_dict.keys())
            
            # Format 2: dict directement embeddings
            else:
                embeddings_dict = emb_data
                patient_ids = list(embeddings_dict.keys())
        
        # Format 3: dict simple {patient_id: embedding}
        elif isinstance(emb_data, dict):
            embeddings_dict = emb_data
            patient_ids = list(embeddings_dict.keys())
        
        else:
            print(f"X Format non reconnu: {type(emb_data)}")
            return None, None
        
        print(f"   - Patients: {len(patient_ids)}")
        print(f"   - Embeddings: {len(embeddings_dict)}")
        
        # Convertir en matrice numpy
        emb_matrix = []
        valid_patient_ids = []
        
        for pid in patient_ids:
            if pid in embeddings_dict:
                emb = embeddings_dict[pid]
                
                # Convertir selon le type
                if isinstance(emb, np.ndarray):
                    emb_matrix.append(emb)
                elif isinstance(emb, (list, tuple)):
                    emb_matrix.append(np.array(emb))
                elif torch.is_tensor(emb):
                    emb_matrix.append(emb.cpu().numpy())
                else:
                    print(f"!!  Type d'embedding non supporté pour {pid}: {type(emb)}")
                    continue
                
                valid_patient_ids.append(pid)
        
        emb_matrix = np.array(emb_matrix)
        print(f"vv Matrice shape: {emb_matrix.shape}")
        
        return emb_matrix, valid_patient_ids
        
    except FileNotFoundError:
        print(f"XX Fichier {embeddings_path} non trouvé!")
        print("   Exécutez d'abord l'entraînement du modèle")
        return None, None
    except Exception as e:
        print(f"XX Erreur de chargement: {e}")
        return None, None

def load_patient_labels(labels_path='fixed_sequences.pkl'):
    """Charge les labels des patients (pathways)"""
    
    print(f"\n-> Chargement des labels depuis {labels_path}...")
    
    try:
        with open(labels_path, 'rb') as f:
            data = pickle.load(f)
        
        patient_info = data.get('patient_info', {})
        
        # Extraire les pathways
        labels_dict = {}
        for pid, info in patient_info.items():
            if 'Pathway' in info:
                labels_dict[pid] = info['Pathway']
        
        print(f"vv Labels chargés: {len(labels_dict)} patients avec pathways")
        
        return labels_dict
        
    except FileNotFoundError:
        print(f"!!  Fichier {labels_path} non trouvé, clustering sans labels")
        return {}
    except Exception as e:
        print(f"!!  Erreur de chargement labels: {e}")
        return {}

def cluster_medical_embeddings(n_clusters=10, embeddings_path='medical_embeddings.pkl'):
    """Clustering avec les embeddings médicaux"""
    
    print("oo CLUSTERING AVEC EMBEDDINGS MÉDICAUX")
    print("=" * 50)
    
    # 1. Charger les embeddings
    emb_matrix, patient_ids = load_medical_embeddings(embeddings_path)
    
    if emb_matrix is None or patient_ids is None:
        return None
    
    # 2. Charger les labels
    labels_dict = load_patient_labels()
    
    # 3. Aligner labels avec embeddings
    true_labels = []
    valid_indices = []
    
    for i, pid in enumerate(patient_ids):
        if pid in labels_dict:
            true_labels.append(labels_dict[pid])
            valid_indices.append(i)
        else:
            true_labels.append('Unknown')
    
    print(f"\nStats : Patients avec labels: {len(valid_indices)}/{len(patient_ids)}")
    
    # 4. Déterminer le nombre optimal de clusters
    n_samples = len(emb_matrix)
    
    if n_clusters is None:
        # Recherche automatique du K optimal
        print("\nq Recherche du nombre optimal de clusters...")
        
        max_k = min(15, n_samples // 10)
        if max_k < 2:
            max_k = 2
        
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(emb_matrix)
            score = silhouette_score(emb_matrix, labels)
            silhouette_scores.append(score)
            print(f"   K={k}: Silhouette = {score:.3f}")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"vv K optimal: {optimal_k}")
        n_clusters = optimal_k
    
    # 5. Clustering
    print(f"\noo Clustering avec K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=25, max_iter=500)
    cluster_labels = kmeans.fit_predict(emb_matrix)
    
    # 6. Calculer les métriques
    metrics = {}
    
    if len(valid_indices) > 0:
        # Extraire les labels et clusters pour les patients avec labels
        valid_clusters = cluster_labels[valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]
        
        # Encoder les labels textuels
        le = LabelEncoder()
        true_encoded = le.fit_transform(valid_true_labels)
        
        # Calculer métriques
        ari = adjusted_rand_score(true_encoded, valid_clusters)
        nmi = normalized_mutual_info_score(true_encoded, valid_clusters)
        silhouette = silhouette_score(emb_matrix, cluster_labels)
        
        metrics = {'ari': ari, 'nmi': nmi, 'silhouette': silhouette}
        
        print(f"\n RÉSULTATS:")
        print(f"   ARI: {ari:.3f}")
        print(f"   NMI: {nmi:.3f}")
        print(f"   Silhouette Score: {silhouette:.3f}")
    
    # 7. Distribution des clusters
    print(f"\nStats : Distribution des clusters:")
    cluster_counts = Counter(cluster_labels)
    for cluster_id in range(n_clusters):
        count = cluster_counts.get(cluster_id, 0)
        percentage = count / n_samples * 100
        print(f"   Cluster {cluster_id}: {count} patients ({percentage:.1f}%)")
    
    # 8. Analyse par cluster
    if len(valid_indices) > 0:
        print(f"\nq Analyse par cluster:")
        
        for cluster_id in range(n_clusters):
            # Indices des patients dans ce cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Trouver les patients avec labels dans ce cluster
                labeled_in_cluster = []
                for idx in cluster_indices:
                    if idx in valid_indices:
                        labeled_in_cluster.append(true_labels[idx])
                
                if labeled_in_cluster:
                    pathway_counts = Counter(labeled_in_cluster)
                    most_common = pathway_counts.most_common(1)[0]
                    proportion = most_common[1] / len(labeled_in_cluster)
                    
                    print(f"   Cluster {cluster_id} ({len(cluster_indices)} patients):")
                    print(f"      Pathway dominant: {most_common[0]} ({proportion:.1%})")
                    
                    # Afficher les 3 principaux pathways
                    if len(pathway_counts) > 1:
                        print(f"      Top 3: {['{} ({:.1%})'.format(p, c/len(labeled_in_cluster)) for p, c in pathway_counts.most_common(3)]}")
    
    # 9. Visualisation
    print("\n Visualisation...")
    
    if n_samples > 10:
        # t-SNE
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        emb_2d = tsne.fit_transform(emb_matrix)
        
        # Créer figure
        n_plots = 2 if len(valid_indices) > 0 else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot 1: Clusters prédits
        scatter1 = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                  c=cluster_labels, cmap='tab20', alpha=0.7, s=30)
        plt.colorbar(scatter1, ax=axes[0])
        title1 = f'Clusters (K={n_clusters})'
        if 'silhouette' in metrics:
            title1 += f'\nSilhouette: {metrics["silhouette"]:.3f}'
        axes[0].set_title(title1)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        
        # Plot 2: Vrais Pathways si disponibles
        if len(valid_indices) > 0 and n_plots > 1:
            # Créer un array de labels pour tous les patients
            all_labels = np.array(['Unknown'] * n_samples)
            for idx, label in zip(valid_indices, valid_true_labels):
                all_labels[idx] = label
            
            # Encoder pour la couleur
            le_vis = LabelEncoder()
            labels_encoded = le_vis.fit_transform(all_labels)
            
            scatter2 = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                      c=labels_encoded, cmap='tab20', alpha=0.7, s=30)
            plt.colorbar(scatter2, ax=axes[1])
            title2 = f'Vrais Pathways'
            if 'ari' in metrics:
                title2 += f'\nARI: {metrics["ari"]:.3f}'
            axes[1].set_title(title2)
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig('medical_clustering_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 10. Sauvegarder les résultats
    print("\n Sauvegarde des résultats...")
    
    results = {
        'patient_ids': patient_ids,
        'embeddings': emb_matrix,
        'cluster_labels': cluster_labels.tolist(),
        'true_labels': true_labels,
        'n_clusters': n_clusters,
        'metrics': metrics
    }
    
    if n_samples > 10 and 'emb_2d' in locals():
        results['embeddings_2d'] = emb_2d
    
    output_path = 'medical_clustering_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"vv Clustering terminé!")
    print(f" Résultats sauvegardés dans '{output_path}'")
    
    return results

def cluster_with_fixed_k(k=10):
    """Clustering avec K fixé"""
    return cluster_medical_embeddings(n_clusters=k)

def analyze_cluster_quality(results):
    """Analyse détaillée de la qualité des clusters"""
    
    if results is None:
        print("XX Aucun résultat à analyser")
        return
    
    print("\n ANALYSE DÉTAILLÉE DES CLUSTERS")
    print("="*50)
    
    cluster_labels = results['cluster_labels']
    true_labels = results['true_labels']
    
    # Compter combien de clusters sont "purs"
    n_clusters = len(set(cluster_labels))
    pure_clusters = 0
    
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        
        if cluster_indices:
            # Labels dans ce cluster (ignorer 'Unknown')
            labels_in_cluster = [true_labels[i] for i in cluster_indices if true_labels[i] != 'Unknown']
            
            if labels_in_cluster:
                pathway_counts = Counter(labels_in_cluster)
                majority_count = pathway_counts.most_common(1)[0][1]
                purity = majority_count / len(labels_in_cluster)
                
                if purity > 0.7:  # 70% de pureté minimum
                    pure_clusters += 1
                    print(f"✓ Cluster {cluster_id}: {purity:.1%} de pureté")
                else:
                    print(f"✗ Cluster {cluster_id}: seulement {purity:.1%} de pureté")
    
    print(f"\nStats : {pure_clusters}/{n_clusters} clusters avec pureté > 70%")

# Version simple pour test rapide
def quick_clustering():
    """Clustering rapide sans toutes les analyses"""
    
    print("⚡ CLUSTERING RAPIDE")
    
    # Charger embeddings
    emb_matrix, patient_ids = load_medical_embeddings()
    
    if emb_matrix is None:
        return
    
    # Clustering simple avec K=10
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(emb_matrix)
    
    # Visualisation simple
    if len(emb_matrix) > 10:
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(emb_matrix)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=clusters, cmap='tab20', alpha=0.6, s=20)
        plt.colorbar()
        plt.title('Clusters de patients (K=10)')
        plt.tight_layout()
        plt.savefig('quick_clustering.png', dpi=150)
        plt.show()
    
    # Distribution
    print("\nStats : Distribution:")
    for i in range(10):
        count = np.sum(clusters == i)
        print(f"   Cluster {i}: {count} patients")

if __name__ == "__main__":
    print("Options de clustering:")
    print("1. Clustering avec K optimal (recherche automatique)")
    print("2. Clustering avec K=10 (10 parcours)")
    print("3. Clustering rapide (visualisation seulement)")
    print("4. Analyser résultats existants")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == "1":
        results = cluster_medical_embeddings(embeddings_path='medical_embeddings.pkl')
    elif choice == "2":
        results = cluster_with_fixed_k(10)
    elif choice == "3":
        quick_clustering()
        results = None
    elif choice == "4":
        try:
            with open('medical_clustering_results.pkl', 'rb') as f:
                results = pickle.load(f)
            analyze_cluster_quality(results)
        except FileNotFoundError:
            print("XX Fichier de résultats non trouvé")
            results = None
    else:
        print("XX Choix invalide")
        results = None
    
    if results is not None:
        print(f"\nvv Clustering terminé avec succès!")
        print(f"   Patients: {len(results['patient_ids'])}")
        print(f"   Clusters: {results['n_clusters']}")
        
        if 'metrics' in results and results['metrics']:
            metrics = results['metrics']
            print(f"   ARI: {metrics.get('ari', 'N/A'):.3f}")
            print(f"   NMI: {metrics.get('nmi', 'N/A'):.3f}")