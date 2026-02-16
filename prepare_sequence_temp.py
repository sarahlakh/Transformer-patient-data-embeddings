import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from collections import defaultdict
import time

def prepare_sequences_with_temporal(df_char_path, bdd_path, output_path='patient_sequences_temporal.pkl'):
    """
    Crée des séquences de patients avec encodage temporel avancé
    Prend en compte:
    1. L'ordre des codes DANS une visite
    2. L'ordre chronologique DES visites
    3. Les intervalles de temps entre visites
    4. La temporalité relative (1ère visite, etc.)
    """
    start_time = time.time()
    
    # 1. Chargement
    print("1. Chargement des données...")
    df_char = pd.read_csv(df_char_path)
    with open(bdd_path, 'rb') as f:
        df_visits = pickle.load(f)
    
    print(f"   - {len(df_char)} patients dans df_char")
    print(f"   - {len(df_visits)} lignes de visites dans Bdd.pkl")
    
    # 2. Renommage et conversion des dates
    print("2. Prétraitement des dates...")
    df_visits.rename(columns={'BEN_IDT_ANO': 'ID_PATIENT'}, inplace=True)
    df_visits['DATE'] = pd.to_datetime(df_visits['DATE'])
    
    # 3. TRI CRITIQUE : Trier une seule fois par patient et date
    print("3. Tri des visites...")
    df_visits.sort_values(['ID_PATIENT', 'DATE'], inplace=True)
    
    # 4. Grouper les visites par patient
    print("4. Regroupement des visites par patient...")
    grouped = df_visits.groupby('ID_PATIENT')
    
    # 5. Préparer les structures de résultats
    patient_sequences = {}
    pathway_labels = {}
    temporal_info = {}  # Nouveau: stocke les infos temporelles
    
    # 6. Convertir df_char en dict pour accès O(1)
    print("5. Création du dictionnaire patients...")
    patient_info = df_char.set_index('ID_PATIENT')[['Pathway', 'AGE_DIAG', 'BC_SubType']].to_dict('index')
    
    # 7. Traiter chaque patient avec encodage temporel
    print("6. Construction des séquences avec encodage temporel...")
    processed = 0
    skipped_no_visits = 0
    skipped_no_year = 0
    
    for patient_id, visits in grouped:
        processed += 1
        if processed % 1000 == 0:
            print(f"   Traités: {processed}/{len(grouped)} patients")
        
        # Vérifier si le patient existe dans df_char
        if patient_id not in patient_info:
            skipped_no_visits += 1
            continue
        
        # Prendre les visites de la première année
        start_date = visits['DATE'].iloc[0]
        end_date = start_date + timedelta(days=365)
        year_visits = visits[visits['DATE'] <= end_date]
        
        if len(year_visits) == 0:
            skipped_no_year += 1
            continue
        
        # Initialiser les structures pour ce patient
        visit_vectors = []
        time_intervals = []  # Jours depuis visite précédente
        visit_positions = []  # Position dans la séquence (1ère, 2ème, etc.)
        absolute_dates = []  # Date absolue
        
        # Parcourir les visites chronologiquement
        prev_date = None
        for visit_idx, (_, visit) in enumerate(year_visits.iterrows(), 1):
            current_date = visit['DATE']
            
            # 1. Encoder la visite (GARDER L'ORDRE DES CODES)
            codes_with_type = []
            
            # ORDRE FIXE IMPORTANT: CCAM → ICD10 → CIP → UC
            # Cet ordre reflète une hiérarchie logique
            if pd.notna(visit['COD_CCAM']):
                codes_with_type.append(f"CCAM:{visit['COD_CCAM']}")
            if pd.notna(visit['COD_ICD10']):
                codes_with_type.append(f"ICD:{visit['COD_ICD10']}")
            if pd.notna(visit['COD_CIP']):
                codes_with_type.append(f"CIP:{visit['COD_CIP']}")
            if pd.notna(visit['COD_UCD']):  # Note: correction de COD_UC à COD_UCD
                codes_with_type.append(f"UC:{visit['COD_UCD']}")
            
            # NE PAS TRIER ! Garder l'ordre fixe défini ci-dessus
            visit_representation = '|'.join(codes_with_type) if codes_with_type else 'NO_CODE'
            
            # 2. Ajouter information temporelle
            temporal_suffix = ""
            
            # a) Intervalle depuis dernière visite
            if prev_date is not None:
                days_since_last = (current_date - prev_date).days
                time_intervals.append(days_since_last)
                
                # Encoder l'intervalle en catégorie
                if days_since_last <= 7:
                    interval_cat = "WEEKLY"
                elif days_since_last <= 30:
                    interval_cat = "MONTHLY"
                elif days_since_last <= 90:
                    interval_cat = "QUARTERLY"
                else:
                    interval_cat = "SPORADIC"
                
                temporal_suffix += f"_INTERVAL:{interval_cat}"
            else:
                time_intervals.append(0)  # Première visite
                temporal_suffix += "_FIRST"
            
            # b) Position dans la séquence
            visit_positions.append(visit_idx)
            
            # Position relative (début/milieu/fin)
            total_visits = len(year_visits)
            position_ratio = visit_idx / total_visits
            
            if position_ratio <= 0.33:
                position_cat = "EARLY"
            elif position_ratio <= 0.67:
                position_cat = "MID"
            else:
                position_cat = "LATE"
            
            temporal_suffix += f"_POS:{position_cat}"
            
            # c) Mois de l'année (saisonnalité)
            month = current_date.month
            if month in [12, 1, 2]:
                season = "WINTER"
            elif month in [3, 4, 5]:
                season = "SPRING"
            elif month in [6, 7, 8]:
                season = "SUMMER"
            else:
                season = "FALL"
            
            temporal_suffix += f"_SEASON:{season}"
            
            # 3. Combiner visite + info temporelle
            enhanced_visit = f"{visit_representation}{temporal_suffix}"
            visit_vectors.append(enhanced_visit)
            
            # 4. Stocker date absolue pour analyses ultérieures
            absolute_dates.append(current_date)
            
            # Mettre à jour pour visite suivante
            prev_date = current_date
        
        # Stocker les séquences avec infos temporelles
        patient_sequences[patient_id] = visit_vectors
        pathway_labels[patient_id] = patient_info[patient_id]['Pathway']
        
        # Stocker les infos temporelles détaillées
        temporal_info[patient_id] = {
            'time_intervals': time_intervals,
            'visit_positions': visit_positions,
            'absolute_dates': absolute_dates,
            'total_visits': len(year_visits),
            'duration_days': (year_visits['DATE'].iloc[-1] - start_date).days if len(year_visits) > 1 else 0
        }
    
    # 8. Calculer des statistiques temporelles globales
    print("\n7. Calcul des statistiques temporelles...")
    
    all_intervals = []
    for intervals in temporal_info.values():
        all_intervals.extend(intervals['time_intervals'][1:])  # Exclure le 0 de la première visite
    
    if all_intervals:
        print(f"   - Intervalle moyen entre visites: {np.mean(all_intervals):.1f} jours")
        print(f"   - Intervalle médian: {np.median(all_intervals):.1f} jours")
        print(f"   - Nombre moyen de visites/an: {np.mean([len(v) for v in patient_sequences.values()]):.1f}")
    
    # 9. Créer un vocabulaire temporel enrichi
    print("\n8. Création du vocabulaire temporel...")
    
    # Extraire tous les tokens uniques
    all_tokens = set()
    for seq in patient_sequences.values():
        for visit in seq:
            tokens = visit.split('_')
            for token in tokens:
                all_tokens.add(token)
    
    print(f"   - Tokens uniques totaux: {len(all_tokens)}")
    
    # 10. Sauvegarder avec toutes les infos
    print("\n9. Sauvegarde...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'sequences': patient_sequences, 
            'labels': pathway_labels,
            'patient_info': patient_info,
            'temporal_info': temporal_info,
            'statistics': {
                'mean_interval': np.mean(all_intervals) if all_intervals else 0,
                'median_interval': np.median(all_intervals) if all_intervals else 0,
                'mean_visits': np.mean([len(v) for v in patient_sequences.values()]),
                'unique_tokens': len(all_tokens)
            }
        }, f)
    
    total_time = time.time() - start_time
    print(f"\n✅ Terminé en {total_time:.1f} secondes ({total_time/60:.1f} minutes)")
    print(f"📊 Statistiques:")
    print(f"   - Patients avec séquences: {len(patient_sequences)}")
    print(f"   - Patients sans visites dans df_char: {skipped_no_visits}")
    print(f"   - Patients sans visites sur 1 an: {skipped_no_year}")
    
    # Afficher un exemple enrichi
    if patient_sequences:
        first_pid = list(patient_sequences.keys())[0]
        print(f"\n📋 EXEMPLE AVEC ENCODAGE TEMPOREL (Patient {first_pid}):")
        print(f"   Pathway: {pathway_labels[first_pid]}")
        print(f"   Nombre de visites: {len(patient_sequences[first_pid])}")
        print(f"   Visites encodées:")
        for i, visit in enumerate(patient_sequences[first_pid][:3]):
            print(f"     {i+1}. {visit}")
        
        # Afficher les intervalles
        if first_pid in temporal_info:
            intervals = temporal_info[first_pid]['time_intervals']
            if len(intervals) > 1:
                print(f"   Intervalles entre visites: {intervals[1:]} jours")
    
    return patient_sequences, pathway_labels, temporal_info

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
    patient_seqs, labels, temporal = prepare_sequences_with_temporal(
        df_char_path='df_char(1).csv',
        bdd_path='Bdd.pkl',
        output_path='patient_sequences_temporal.pkl'
    )
    print("OPTION 1: Séquences médicales pures")
    medical_seqs = extract_pure_medical_sequences(
        input_path="patient_sequences_temporal.pkl",
        output_path="medical_sequences_pure.pkl",
        max_visits=30
    )
    
    # Option 2: Analyse rapide
    print("\n" + "="*50)
    print("OPTION 2: Analyse des séquences")
    analyze_medical_sequences("medical_sequences_pure.pkl")