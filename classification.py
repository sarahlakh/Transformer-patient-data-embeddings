import numpy as np
import pickle

def simple_pathway_classification():
    """Test de classification simple pour voir si c'est possible"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Créer des features basiques
    with open('medical_sequences_pure.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Encoder les séquences en bag-of-words
    all_codes = set()
    for seq in data['sequences'].values():
        all_codes.update(seq)
    
    code_to_idx = {code: i for i, code in enumerate(all_codes)}
    
    # Créer la matrice de features
    X = []
    y = []
    
    for pid, seq in data['sequences'].items():
        if pid in data['patient_info']:
            pathway = data['patient_info'][pid].get('Pathway')
            if pathway:
                # Bag-of-words
                features = np.zeros(len(all_codes))
                for code in seq[:20]:  # 20 premiers codes
                    if code in code_to_idx:
                        features[code_to_idx[code]] += 1
                
                X.append(features)
                y.append(pathway)
    
    X = np.array(X)
    y = np.array(y)
    
    # Classification simple
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Accuracy RandomForest: {accuracy:.3f}")
    
    return accuracy

if __name__ == "__main__":
    accuracy = simple_pathway_classification()
    print(accuracy)