# explore_pickle.py
import pickle
import numpy as np
import pandas as pd
from collections import Counter

def explore_pickle(file_path, max_items=5):
    """Explore un fichier pickle en profondeur"""
    
    print(f"\n🔍 EXPLORATION DE: {file_path}")
    print("="*60)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type de données: {type(data)}")
        
        # Si c'est un dictionnaire
        if isinstance(data, dict):
            print(f"Clés: {list(data.keys())}")
            print(f"Nombre d'éléments: {len(data)}")
            
            for key, value in data.items():
                print(f"\n📌 Clé: '{key}'")
                print(f"   Type: {type(value)}")
                
                if isinstance(value, (list, tuple)):
                    print(f"   Taille: {len(value)}")
                    if len(value) > 0:
                        print(f"   Premier élément: {value[0]}")
                        print(f"   Type élément: {type(value[0])}")
                
                elif isinstance(value, dict):
                    print(f"   Nombre de sous-éléments: {len(value)}")
                    if len(value) > 0:
                        first_key = next(iter(value.keys()))
                        print(f"   Exemple clé: {first_key}")
                        print(f"   Exemple valeur: {value[first_key]}")
                
                elif isinstance(value, np.ndarray):
                    print(f"   Shape: {value.shape}")
                    print(f"   Dtype: {value.dtype}")
                    print(f"   Valeurs min/max: {value.min():.3f}, {value.max():.3f}")
                
                elif hasattr(value, 'shape'):  # Pour torch.Tensor
                    print(f"   Shape: {value.shape}")
                
                else:
                    # Afficher juste un aperçu
                    value_str = str(value)
                    if len(value_str) > 100:
                        print(f"   Valeur: {value_str[:100]}...")
                    else:
                        print(f"   Valeur: {value}")
                
                # Limiter l'affichage
                if list(data.keys()).index(key) >= max_items - 1:
                    print(f"\n[... {len(data) - max_items} autres clés ...]")
                    break
        
        # Si c'est une liste
        elif isinstance(data, list):
            print(f"Taille: {len(data)}")
            if len(data) > 0:
                print(f"Premier élément: {data[0]}")
                print(f"Type premier élément: {type(data[0])}")
        
        # Si c'est un numpy array
        elif isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Exemple valeurs: {data[:3] if len(data) > 3 else data}")
        
        else:
            print(f"Valeur: {data}")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")

# Explorer les fichiers principaux
files_to_explore = [
    'medical_sequences_pure.pkl',
    'medical_embeddings.pkl',
    'pathway_embeddings_from_rf.pkl',
    'fixed_sequences.pkl',
    'pathway_features_fixed.pkl'
]

for file in files_to_explore:
    explore_pickle(file)