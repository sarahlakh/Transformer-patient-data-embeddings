# create_pathway_features_fixed.py
import pickle
import numpy as np
from collections import Counter, defaultdict
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

def create_fixed_length_features():
    """Crée des features de longueur fixe pour chaque patient"""
    
    print("🧠 CRÉATION DE FEATURES DE LONGUEUR FIXE")
    print("="*50)
    
    # Charger les données
    with open('medical_sequences_pure.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    patient_info = data.get('patient_info', {})
    
    print(f"Données chargées: {len(sequences)} patients")
    
    # 1. Identifier les codes les plus discriminants
    print("\n🔍 Identification des codes discriminants...")
    
    # Analyser la fréquence par pathway
    pathway_code_freq = defaultdict(Counter)
    
    for pid, seq in sequences.items():
        if pid in patient_info and 'Pathway' in patient_info[pid]:
            pathway = patient_info[pid]['Pathway']
            for code in seq[:20]:  # Limiter aux 20 premiers codes
                pathway_code_freq[pathway][code] += 1
    
    # Trouver les codes spécifiques à chaque pathway
    all_codes = set()
    for pathway in pathway_code_freq:
        all_codes.update(pathway_code_freq[pathway].keys())
    
    print(f"Codes uniques totaux: {len(all_codes)}")
    
    # 2. Sélectionner les codes les plus discriminants
    n_top_codes = 100  # Nombre de codes à utiliser comme features
    code_importance = {}
    
    for code in all_codes:
        # Calculer l'entropie de la distribution sur les pathways
        counts = []
        total = 0
        for pathway in range(1, 11):
            count = pathway_code_freq[pathway].get(code, 0)
            counts.append(count)
            total += count
        
        if total > 0:
            # Normaliser
            probs = [c/total for c in counts]
            # Éviter log(0)
            probs = [p if p > 0 else 1e-10 for p in probs]
            # Calculer entropie
            entropy = -sum(p * np.log(p) for p in probs)
            code_importance[code] = entropy
    
    # Les codes les plus discriminants ont une entropie faible
    sorted_codes = sorted(code_importance.items(), key=lambda x: x[1])
    top_codes = [code for code, entropy in sorted_codes[:n_top_codes]]
    
    print(f"Top {len(top_codes)} codes discriminants sélectionnés")
    
    # 3. Créer des features de longueur fixe
    print("\n📊 Création des vecteurs de features...")
    
    # Définir les features
    feature_names = []
    feature_vectors = []
    patient_ids = []
    pathways = []
    
    # Features basiques (toujours présentes)
    base_features = [
        'seq_length',  # Longueur de la séquence
        'unique_codes',  # Nombre de codes uniques
        'has_C50',  # Contient ICD:C50
        'has_Z511',  # Contient ICD:Z511
        'has_Z5100',  # Contient ICD:Z5100
        'has_Z5101',  # Contient ICD:Z5101
        'has_ZZLF900',  # Contient CCAM:ZZLF900
        'has_NO_CODE',  # Contient NO_CODE
    ]
    
    # Ajouter les codes discriminants
    for code in top_codes:
        # Simplifier le nom pour la feature
        simple_name = code.replace(':', '_').replace('.', '_')
        base_features.append(f'has_{simple_name}')
    
    # Ajouter des features de position
    for pos in range(5):  # 5 premières positions
        base_features.append(f'pos{pos}_code')
    
    feature_names = base_features.copy()
    
    # 4. Remplir les features pour chaque patient
    print("   Création des vecteurs...")
    
    for pid, seq in sequences.items():
        if pid not in patient_info:
            continue
        
        pathway = patient_info[pid].get('Pathway')
        if not pathway:
            continue
        
        # Initialiser le vecteur de features
        features = []
        
        # 1. Features basiques
        features.append(len(seq))  # seq_length
        features.append(len(set(seq)))  # unique_codes
        
        # 2. Présence de codes importants
        features.append(1 if 'ICD:C50' in seq else 0)  # has_C50
        features.append(1 if 'ICD:Z511' in seq else 0)  # has_Z511
        features.append(1 if 'ICD:Z5100' in seq else 0)  # has_Z5100
        features.append(1 if 'ICD:Z5101' in seq else 0)  # has_Z5101
        features.append(1 if 'CCAM:ZZLF900' in seq else 0)  # has_ZZLF900
        features.append(1 if 'NO_CODE' in seq else 0)  # has_NO_CODE
        
        # 3. Présence des codes discriminants
        for code in top_codes:
            features.append(1 if code in seq else 0)
        
        # 4. Codes aux premières positions (encodés)
        for pos in range(5):
            if pos < len(seq):
                # Encoder le code à cette position
                code = seq[pos]
                # Utiliser un hash simple
                code_hash = hash(code) % 100
                features.append(code_hash)
            else:
                features.append(-1)  # Valeur pour position vide
        
        # Vérifier que toutes les features sont des nombres
        features = [float(f) for f in features]
        
        feature_vectors.append(features)
        patient_ids.append(pid)
        pathways.append(pathway)
    
    # Convertir en numpy arrays
    X = np.array(feature_vectors)
    y = np.array(pathways)
    
    print(f"✅ Features créées: {X.shape[0]} patients, {X.shape[1]} features")
    
    # 5. Sauvegarder
    feature_data = {
        'X': X,
        'y': y,
        'patient_ids': patient_ids,
        'feature_names': feature_names,
        'top_codes': top_codes
    }
    
    with open('pathway_features_fixed.pkl', 'wb') as f:
        pickle.dump(feature_data, f)
    
    print(f"📁 Features sauvegardées dans 'pathway_features_fixed.pkl'")
    
    return feature_data

def test_with_randomforest(feature_data):
    """Teste les features avec RandomForest"""
    
    print("\n🧪 TEST AVEC RANDOMFOREST")
    print("="*50)
    
    X = feature_data['X']
    y = feature_data['y']
    
    print(f"Données: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"Distribution des pathways: {Counter(y)}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Entraîner RandomForest
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Important pour données déséquilibrées
    )
    
    print("Entraînement en cours...")
    clf.fit(X_train, y_train)
    
    # Prédictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📈 RÉSULTATS:")
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Matrice de confusion
    from sklearn.metrics import confusion_matrix, classification_report
    
    print(f"\n📊 Matrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Afficher en format plus lisible
    pathways = sorted(set(y))
    print("   Vrai\\Prédit:", end="")
    for p in pathways:
        print(f"{p:>6}", end="")
    print()
    
    for i, true_p in enumerate(pathways):
        print(f"   {true_p:>11}", end="")
        for j, pred_p in enumerate(pathways):
            print(f"{cm[i, j]:>6}", end="")
        print()
    
    # Rapport de classification
    print(f"\n📋 Rapport de classification:")
    report = classification_report(y_test, y_pred, digits=3)
    print(report)
    
    # Importance des features
    print(f"\n🔍 TOP 20 FEATURES IMPORTANTES:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    for idx in indices:
        if idx < len(feature_data['feature_names']):
            feat_name = feature_data['feature_names'][idx]
            print(f"   {feat_name}: {importances[idx]:.4f}")
    
    return accuracy, clf

def create_embeddings_from_features(feature_data, clf):
    """Crée des embeddings à partir des features et du modèle"""
    
    print("\n🧬 CRÉATION D'EMBEDDINGS")
    print("="*50)
    
    X = feature_data['X']
    patient_ids = feature_data['patient_ids']
    
    # Utiliser les prédictions probabilistes comme embeddings
    print("Calcul des probabilités...")
    probabilities = clf.predict_proba(X)
    
    # Créer le dict d'embeddings
    embeddings_dict = {}
    for i, pid in enumerate(patient_ids):
        embeddings_dict[pid] = probabilities[i]
    
    print(f"Embeddings créés: {len(embeddings_dict)} patients")
    print(f"Dimension embeddings: {probabilities.shape[1]}")
    
    # Sauvegarder
    embeddings_data = {
        'embeddings': embeddings_dict,
        'patient_ids': patient_ids,
        'pathways': feature_data['y'].tolist(),
        'feature_importances': clf.feature_importances_.tolist()
    }
    
    with open('pathway_embeddings_from_rf.pkl', 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print("📁 Embeddings sauvegardés dans 'pathway_embeddings_from_rf.pkl'")
    
    return embeddings_dict

def cluster_pathway_embeddings():
    """Clustering sur les nouveaux embeddings avec Davies-Bouldin"""
    
    print("\n🎯 CLUSTERING SUR NOUVEAUX EMBEDDINGS")
    print("="*50)
    
    # Charger les embeddings
    with open('pathway_embeddings_from_rf.pkl', 'rb') as f:
        embeddings_data = pickle.load(f)
    
    embeddings_dict = embeddings_data['embeddings']
    patient_ids = embeddings_data['patient_ids']
    pathways = embeddings_data['pathways']
    
    # Convertir en matrice
    emb_matrix = np.array([embeddings_dict[pid] for pid in patient_ids])
    
    print(f"Embeddings shape: {emb_matrix.shape}")
    
    # Clustering avec différents K pour trouver l'optimal
    from sklearn.cluster import KMeans
    
    print("\n🔍 Recherche du K optimal...")
    
    # Tester différents K
    k_range = range(2, min(16, emb_matrix.shape[0] // 10 + 1))
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=25, max_iter=300)
        cluster_labels = kmeans.fit_predict(emb_matrix)
        
        # Calculer les scores
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(emb_matrix, cluster_labels)
        db_score = davies_bouldin_score(emb_matrix, cluster_labels)
        
        silhouette_scores.append(sil_score)
        davies_bouldin_scores.append(db_score)
        
        print(f"   K={k}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}")
    
    # Meilleur K selon Silhouette (plus haut = mieux)
    best_k_sil = k_range[np.argmax(silhouette_scores)]
    # Meilleur K selon Davies-Bouldin (plus bas = mieux)
    best_k_db = k_range[np.argmin(davies_bouldin_scores)]
    
    print(f"\n📊 Meilleur K (Silhouette): {best_k_sil}")
    print(f"📊 Meilleur K (Davies-Bouldin): {best_k_db}")
    
    # Choisir le K final (compromis)
    if best_k_sil == best_k_db:
        optimal_k = best_k_sil
        print(f"\n✅ Consensus: K optimal = {optimal_k}")
    else:
        # Prendre la moyenne ou le K recommandé par Silhouette
        optimal_k = best_k_sil
        print(f"\n⚠️  Désaccord entre métriques, utilisation de K={optimal_k} (basé sur Silhouette)")
    
    # Clustering final avec K optimal
    print(f"\n🔨 Clustering final avec K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=25)
    cluster_labels = kmeans.fit_predict(emb_matrix)
    
    # Évaluation finale
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(pathways, cluster_labels)
    nmi = normalized_mutual_info_score(pathways, cluster_labels)
    sil_final = silhouette_score(emb_matrix, cluster_labels)
    db_final = davies_bouldin_score(emb_matrix, cluster_labels)
    
    print(f"\n📈 RÉSULTATS FINAUX CLUSTERING:")
    print(f"   ARI: {ari:.3f}")
    print(f"   NMI: {nmi:.3f}")
    print(f"   Silhouette Score: {sil_final:.3f}")
    print(f"   Davies-Bouldin Index: {db_final:.3f}")
    
    # Analyser la correspondance clusters ↔ pathways
    from collections import Counter
    
    print(f"\n🔍 CORRESPONDANCE CLUSTERS ↔ PATHWAYS:")
    
    for cluster_id in range(optimal_k):
        # Indices des patients dans ce cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) > 0:
            # Pathways dans ce cluster
            cluster_pathways = [pathways[i] for i in cluster_indices]
            pathway_counts = Counter(cluster_pathways)
            
            print(f"\n   Cluster {cluster_id} ({len(cluster_indices)} patients):")
            for pathway, count in pathway_counts.most_common(3):
                proportion = count / len(cluster_indices)
                print(f"      Pathway {pathway}: {count} ({proportion:.1%})")
    
    # Sauvegarder les résultats
    results = {
        'k_range': list(k_range),
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_k': optimal_k,
        'cluster_labels': cluster_labels.tolist(),
        'ari': ari,
        'nmi': nmi,
        'silhouette': sil_final,
        'davies_bouldin': db_final
    }
    
    with open('rf_clustering_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n📁 Résultats sauvegardés dans 'rf_clustering_results.pkl'")
    
    return cluster_labels

# Pipeline complet
def main():
    print("🚀 PIPELINE COMPLET POUR AMÉLIORER LE CLUSTERING")
    print("="*60)
    
    # 1. Créer des features de longueur fixe
    print("\n1. 📊 Création des features...")
    feature_data = create_fixed_length_features()
    
    # 2. Tester avec RandomForest
    print("\n2. 🧪 Test avec RandomForest...")
    accuracy, clf = test_with_randomforest(feature_data)
    
    # 3. Créer des embeddings à partir du modèle
    print("\n3. 🧬 Création des embeddings...")
    embeddings_dict = create_embeddings_from_features(feature_data, clf)
    
    # 4. Clustering avec Davies-Bouldin
    print("\n4. 🎯 Clustering avec analyse Davies-Bouldin...")
    cluster_labels = cluster_pathway_embeddings()
    
    print(f"\n" + "="*60)
    print(f"✅ PIPELINE TERMINÉ!")
    
    if accuracy > 0.7:
        print("🎉 Bonne discrimination obtenue!")
        print("➡️  Les nouveaux embeddings devraient donner de meilleurs clusters")
    else:
        print("⚠️  Discrimination moyenne")
        print("➡️  Considérez ajouter plus de features ou utiliser un autre modèle")

# Version simple pour test rapide
def quick_test():
    """Test rapide sans toutes les étapes"""
    
    print("⚡ TEST RAPIDE")
    
    # Créer features simples
    with open('medical_sequences_pure.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    patient_info = data.get('patient_info', {})
    
    # Features très simples
    X = []
    y = []
    patient_ids = []
    
    for pid, seq in sequences.items():
        if pid in patient_info and 'Pathway' in patient_info[pid]:
            pathway = patient_info[pid]['Pathway']
            
            # 5 features simples
            features = [
                len(seq),  # Longueur
                1 if 'ICD:C50' in seq else 0,  # Cancer sein
                1 if 'ICD:Z511' in seq else 0,  # Chimiothérapie
                1 if 'CCAM:ZZLF900' in seq else 0,  # Acte technique
                1 if 'NO_CODE' in seq else 0,  # Trous
            ]
            
            X.append(features)
            y.append(pathway)
            patient_ids.append(pid)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Features créées: {X.shape}")
    
    # RandomForest rapide
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy avec 5 features simples: {accuracy:.3f}")
    
    return accuracy

if __name__ == "__main__":
    print("Options:")
    print("1. Pipeline complet (avec Davies-Bouldin)")
    print("2. Test rapide avec 5 features")
    print("3. Juste créer les features")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        accuracy = quick_test()
        if accuracy > 0.7:
            print("\n✅ Bon départ! Essayez le pipeline complet.")
        else:
            print("\n⚠️  Features trop simples. Essayez avec plus de features.")
    elif choice == "3":
        feature_data = create_fixed_length_features()
        print(f"\n✅ Features créées: {feature_data['X'].shape}")
    else:
        print("❌ Choix invalide")