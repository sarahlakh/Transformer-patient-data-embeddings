import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from collections import defaultdict
import time

def fix_existing_sequences(input_path, output_path):
    """Corrige les séquences existantes en séparant médical/temporel"""
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    fixed_sequences = {}
    
    for pid, seq in data['sequences'].items():
        fixed_seq = []
        
        for token in seq:
            if token == 'NO_CODE':
                fixed_seq.append('NO_CODE')
            else:
                # Séparer médical et temporel
                parts = token.split('_')
                
                # Trouver la partie médicale (avant le premier _INTERVAL:, _POS:, etc.)
                medical_parts = []
                temporal_parts = []
                
                for part in parts:
                    if any(x in part for x in ['INTERVAL:', 'POS:', 'SEASON:', 'WEEKDAY']):
                        temporal_parts.append(part)
                    else:
                        medical_parts.append(part)
                
                # Reconstruire
                if medical_parts:
                    medical = '_'.join(medical_parts)
                    temporal = '|'.join(temporal_parts) if temporal_parts else ''
                    
                    if temporal:
                        fixed_seq.append(f"{medical}_TIME:{temporal}")
                    else:
                        fixed_seq.append(medical)
                else:
                    fixed_seq.append('NO_CODE')
        
        fixed_sequences[pid] = fixed_seq
    
    # Sauvegarder
    with open(output_path, 'wb') as f:
        pickle.dump({
            'sequences': fixed_sequences,
            'patient_info': data['patient_info'],
            'temporal_features': data.get('temporal_features', {})
        }, f)

# extract_medical_sequences.py
import pickle
import re
from collections import Counter

def extract_pure_medical_sequences(input_path, output_path, max_visits=30):
    """
    Extrait uniquement les codes médicaux purs, dans l'ordre
    Format: ['ICD:Z5101', 'CCAM:ZZNL065', 'ICD:C50', ...]
    """
    
    print("🎯 EXTRACTION DES SÉQUENCES MÉDICALES PURES")
    print("="*50)
    
    # 1. Charger les données
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data.get('sequences', {})
    patient_info = data.get('patient_info', {})
    
    print(f"📊 Données chargées: {len(sequences)} patients")
    
    # 2. Extraire les codes médicaux
    medical_sequences = {}
    code_stats = Counter()
    
    for pid, seq in sequences.items():
        medical_seq = []
        
        for token in seq[:max_visits]:  # Limiter le nombre de visites
            if token == 'NO_CODE':
                continue
            
            # Extraire le code médical PUR
            medical_code = extract_pure_medical_code(token)
            if medical_code:
                medical_seq.append(medical_code)
                code_stats[medical_code] += 1
        
        # Garder seulement les séquences avec au moins 3 codes
        if len(medical_seq) >= 3:
            medical_sequences[pid] = medical_seq
    
    print(f"\n📈 STATISTIQUES:")
    print(f"   - Patients avec séquences valides: {len(medical_sequences)}")
    print(f"   - Codes médicaux uniques: {len(code_stats)}")
    
    # 3. Analyser la fréquence des codes
    print(f"\n🔤 TOP 20 CODES MÉDICAUX:")
    total_codes = sum(code_stats.values())
    for code, count in code_stats.most_common(20):
        percentage = count / total_codes * 100
        print(f"   {code}: {count} ({percentage:.1f}%)")
    
    # 4. Analyser par parcours
    print(f"\n🛣️  ANALYSE PAR PARCOURS:")
    pathway_stats = {}
    
    for pid, seq in medical_sequences.items():
        if pid in patient_info and 'Pathway' in patient_info[pid]:
            pathway = patient_info[pid]['Pathway']
            if pathway not in pathway_stats:
                pathway_stats[pathway] = Counter()
            
            # Ajouter les codes de ce patient
            for code in seq:
                pathway_stats[pathway][code] += 1
    
    # Afficher les codes spécifiques à chaque parcours
    for pathway in sorted(pathway_stats.keys()):
        codes = pathway_stats[pathway]
        total = sum(codes.values())
        print(f"\n   Pathway {pathway} ({total} codes):")
        # Afficher les 3 codes les plus fréquents
        for code, count in codes.most_common(3):
            print(f"      {code}: {count} ({count/total:.1%})")
    
    # 5. Sauvegarder
    output_data = {
        'sequences': medical_sequences,
        'patient_info': patient_info,
        'code_vocabulary': list(code_stats.keys()),
        'code_stats': dict(code_stats),
        'config': {
            'max_visits': max_visits,
            'total_patients': len(medical_sequences),
            'total_codes': total_codes
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✅ Données sauvegardées dans {output_path}")
    print(f"📝 Exemple de séquence:")
    sample_pid = next(iter(medical_sequences.keys()))
    print(f"   Patient {sample_pid}: {medical_sequences[sample_pid][:10]}")
    
    return medical_sequences

def extract_pure_medical_code(token):
    """Extrait uniquement le code médical sans informations temporelles"""
    
    # Méthode 1: Si le token a déjà la structure CODE_TIME:...
    if '_TIME:' in token:
        return token.split('_TIME:')[0]
    
    # Méthode 2: Chercher les patterns de codes médicaux
    patterns = [
        # Codes ICD (ex: ICD:Z5101, ICD:C50)
        r'ICD:[A-Z]\d+(?:\.\d+)?',
        # Codes CCAM (ex: CCAM:ZZNL065)
        r'CCAM:[A-Z]+\d+',
        # Codes ATC (ex: ATC:L01XE)
        r'ATC:[A-Z]\d+[A-Z]?',
        # Codes CIM (ex: CIM:C50)
        r'CIM:[A-Z]\d+',
        # Autres codes médicaux courants
        r'[A-Z]{2,4}:\w+'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, token)
        if match:
            return match.group(0)
    
    return None

# Version encore plus simple: avec position implicite
def extract_medical_with_position(input_path, output_path):
    """
    Version qui garde l'ordre mais sans token de position explicite
    L'ordre est implicite dans la séquence
    """
    
    print("🎯 SÉQUENCES MÉDICALES AVEC ORDRE IMPLICITE")
    print("="*50)
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = {}
    
    for pid, seq in data.get('sequences', {}).items():
        clean_seq = []
        
        for token in seq:
            if token == 'NO_CODE':
                continue
            
            code = extract_pure_medical_code(token)
            if code:
                clean_seq.append(code)
        
        # Garder au moins 2 codes
        if len(clean_seq) >= 2:
            sequences[pid] = clean_seq
    
    # Analyser la longueur des séquences
    lengths = [len(seq) for seq in sequences.values()]
    
    print(f"\n📊 Analyse des séquences:")
    print(f"   Patients: {len(sequences)}")
    print(f"   Longueur moyenne: {np.mean(lengths):.1f}")
    print(f"   Longueur max: {max(lengths)}")
    print(f"   Longueur min: {min(lengths)}")
    
    # Sauvegarder
    output_data = {
        'sequences': sequences,
        'patient_info': data.get('patient_info', {}),
        'stats': {
            'avg_length': np.mean(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths)
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    return sequences

# Script d'analyse rapide
def analyze_medical_sequences(seq_path):
    """Analyse rapide des séquences médicales"""
    
    with open(seq_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    
    print("🔍 ANALYSE DES SÉQUENCES MÉDICALES")
    print("="*50)
    
    # 1. Distribution des longueurs
    lengths = [len(seq) for seq in sequences.values()]
    print(f"\n📏 Distribution des longueurs:")
    print(f"   Moyenne: {np.mean(lengths):.1f}")
    print(f"   Médiane: {np.median(lengths):.1f}")
    print(f"   Min: {min(lengths)}")
    print(f"   Max: {max(lengths)}")
    
    # 2. Fréquence des codes
    all_codes = []
    for seq in sequences.values():
        all_codes.extend(seq)
    
    code_counts = Counter(all_codes)
    print(f"\n🔤 Fréquence des codes:")
    print(f"   Codes uniques: {len(code_counts)}")
    print(f"   Total occurrences: {len(all_codes)}")
    
    # Top 10 codes
    print(f"\n🏆 Top 10 codes:")
    for code, count in code_counts.most_common(10):
        percentage = count / len(all_codes) * 100
        print(f"   {code}: {count} ({percentage:.1f}%)")
    
    # 3. Analyse par parcours (si disponible)
    if 'patient_info' in data:
        print(f"\n🛣️  Distribution par parcours:")
        
        pathway_counts = {}
        for pid, seq in sequences.items():
            if pid in data['patient_info']:
                pathway = data['patient_info'][pid].get('Pathway')
                if pathway:
                    if pathway not in pathway_counts:
                        pathway_counts[pathway] = []
                    pathway_counts[pathway].append(len(seq))
        
        for pathway in sorted(pathway_counts.keys()):
            avg_len = np.mean(pathway_counts[pathway])
            count = len(pathway_counts[pathway])
            print(f"   Pathway {pathway}: {count} patients, longueur moyenne: {avg_len:.1f}")

if __name__ == "__main__":
    # Option 1: Séquences médicales pures
    print("OPTION 1: Séquences médicales pures")
    medical_seqs = extract_pure_medical_sequences(
        input_path="fixed_sequences.pkl",
        output_path="medical_sequences_pure.pkl",
        max_visits=30
    )
    
    # Option 2: Analyse rapide
    print("\n" + "="*50)
    print("OPTION 2: Analyse des séquences")
    analyze_medical_sequences("medical_sequences_pure.pkl")