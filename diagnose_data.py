# diagnostic_data.py
import pickle
import numpy as np
import pandas as pd
from collections import Counter

def diagnose_data():
    """Diagnostique vos données et tokenisation"""
    
    print("🔍 DIAGNOSTIC DES DONNÉES")
    print("=" * 60)
    
    # 1. Charger les données
    with open('fixed_sequences.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    patient_info = data['patient_info']
    
    print(f"\n📊 STATISTIQUES GÉNÉRALES:")
    print(f"   - Patients totaux: {len(sequences)}")
    print(f"   - Patients avec info: {len(patient_info)}")
    
    # 2. Analyser les séquences
    print("\n📈 ANALYSE DES SÉQUENCES:")
    
    # Longueur des séquences
    seq_lengths = [len(seq) for seq in sequences.values()]
    print(f"   - Visites par patient:")
    print(f"     Moyenne: {np.mean(seq_lengths):.1f}")
    print(f"     Médiane: {np.median(seq_lengths):.1f}")
    print(f"     Min: {np.min(seq_lengths)}")
    print(f"     Max: {np.max(seq_lengths)}")
    
    # 3. Analyser les tokens
    print("\n🔤 ANALYSE DES TOKENS:")
    
    all_tokens = []
    for seq in sequences.values():
        for visit in seq:
            tokens = visit.split('|')
            all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    print(f"   - Tokens uniques: {len(token_counts)}")
    print(f"   - Total tokens: {len(all_tokens)}")
    
    # Top 10 tokens
    print(f"\n   Top 10 tokens les plus fréquents:")
    for token, count in token_counts.most_common(10):
        percentage = count / len(all_tokens) * 100
        print(f"     {token}: {count} ({percentage:.1f}%)")
    
    # 4. Distribution des tokens par patient
    print("\n🎯 DISTRIBUTION TOKENS/PATIENT:")
    
    tokens_per_patient = []
    for pid, seq in list(sequences.items())[:1000]:  # Échantillon
        patient_tokens = set()
        for visit in seq:
            tokens = visit.split('|')
            patient_tokens.update(tokens)
        tokens_per_patient.append(len(patient_tokens))
    
    print(f"   - Tokens uniques par patient (moyenne): {np.mean(tokens_per_patient):.1f}")
    print(f"   - Tokens uniques par patient (max): {np.max(tokens_per_patient)}")
    
    # 5. Analyser les pathways
    print("\n🛣️ ANALYSE DES PATHWAYS:")
    
    pathways = []
    for info in patient_info.values():
        if 'Pathway' in info:
            pathways.append(info['Pathway'])
    
    pathway_counts = Counter(pathways)
    print(f"   - Pathways uniques: {len(pathway_counts)}")
    
    print(f"\n   Distribution des pathways:")
    for pathway, count in pathway_counts.most_common():
        percentage = count / len(pathways) * 100
        print(f"     Pathway {pathway}: {count} patients ({percentage:.1f}%)")
    
    # 6. Vérifier la tokenisation temporelle
    print("\n🕒 ANALYSE TEMPORELLE:")
    
    if 'temporal_features' in data:
        temp_features = data['temporal_features']
        print(f"   - Patients avec features temporelles: {len(temp_features)}")
        
        # Compter les types de features
        all_temp_tokens = []
        for temp_info in temp_features.values():
            if 'temporal_seq' in temp_info:
                for temp_str in temp_info['temporal_seq']:
                    parts = temp_str.split('|')
                    all_temp_tokens.extend(parts)
        
        temp_counts = Counter(all_temp_tokens)
        print(f"   - Features temporelles uniques: {len(temp_counts)}")
        
        print(f"\n   Top 5 features temporelles:")
        for token, count in temp_counts.most_common(5):
            print(f"     {token}: {count}")
    
    # 7. Recommandations
    print("\n💡 RECOMMANDATIONS:")
    
    issues = []
    
    if np.mean(seq_lengths) < 3:
        issues.append("⚠️  Séquences trop courtes (<3 visites en moyenne)")
    
    if len(token_counts) < 100:
        issues.append(f"⚠️  Vocabulaire trop petit ({len(token_counts)} tokens)")
    
    if max(seq_lengths) > 100:
        issues.append(f"⚠️  Certaines séquences très longues ({max(seq_lengths)} visites)")
    
    if len(pathway_counts) < 5:
        issues.append(f"⚠️  Peu de pathways distincts ({len(pathway_counts)})")
    
    if issues:
        print("   Problèmes détectés:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ Données apparemment correctes")
    
    return {
        'n_patients': len(sequences),
        'avg_seq_length': np.mean(seq_lengths),
        'n_tokens': len(token_counts),
        'n_pathways': len(pathway_counts)
    }

if __name__ == "__main__":
    diagnose_data()