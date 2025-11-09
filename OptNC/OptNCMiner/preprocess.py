import logging
import random
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def create_glob_set(path, n_pos=7000, n_neg=7000, test_size=0.1):
    # 1. 모든 CSV 파일 읽기
    files = glob.glob(f"db/raw/{path}/*.csv")
    all_data = []
    
    for file_path in files:
        protein_name = file_path.split('/')[-1].split('\\')[-1].split('.')[0]
        df = pd.read_csv(file_path)
        
        x_cols = [col for col in df.columns if col.startswith('X')]
        target_col = 'potency' if 'potency' in df.columns else 'Y'
        label_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
        
        if target_col not in df.columns or not x_cols:
            continue
            
        df['fn'] = protein_name
        df['class'] = protein_name
        df.loc[df[target_col] == 0, 'class'] = '-'
        
        all_data.append(df)
        logging.info(f"Loaded {protein_name}: {len(df)} compounds")
    
    # 2. 데이터 병합 및 분할
    combined = pd.concat(all_data, ignore_index=True)
    train_df, test_df = train_test_split(combined, test_size=test_size, stratify=combined['class'])
    
    train_df.to_csv(f"db/processed/{path}/sim_train.csv", index=False)
    test_df.to_csv(f"db/processed/{path}/sim_test.csv", index=False)
    
    # 3. 쌍 데이터 생성
    x_cols = [col for col in combined.columns if col.startswith('X')]
    target_col = 'potency' if 'potency' in combined.columns else 'Y'
    label_col = 'SMILES' if 'SMILES' in combined.columns else 'smiles'
    
    pairs = create_pairs(train_df, x_cols, target_col, label_col, n_pos, n_neg)
    pairs.to_csv(f"db/processed/{path}/sim_data.csv", index=False)

def create_pairs(df, x_cols, target_col, label_col, n_pos, n_neg):
    pairs = []
    proteins = df[df[target_col] == 1]['fn'].unique()
    
    # Positive pairs (같은 단백질에 활성)
    for i in range(n_pos):
        protein = random.choice(proteins)
        active_compounds = df[(df['fn'] == protein) & (df[target_col] == 1)]
        
        if len(active_compounds) < 2:
            continue
            
        sample1, sample2 = active_compounds.sample(2).iloc
        combined_features = list(sample1[x_cols]) + list(sample2[x_cols])
        
        pair = {
            'left': sample1[label_col],
            'right': sample2[label_col], 
            'Y': 1
        }
        pair.update({f'X{j+1}': val for j, val in enumerate(combined_features)})
        pairs.append(pair)
    
    # Negative pairs (다른 단백질)
    for i in range(n_neg):
        if len(proteins) < 2:
            continue
            
        p1, p2 = random.sample(list(proteins), 2)
        
        compounds1 = df[(df['fn'] == p1) & (df[target_col] == 1)]
        compounds2 = df[df['fn'] == p2]
        
        if len(compounds1) == 0 or len(compounds2) == 0:
            continue
            
        sample1 = compounds1.sample(1).iloc[0]
        sample2 = compounds2.sample(1).iloc[0]
        combined_features = list(sample1[x_cols]) + list(sample2[x_cols])
        
        pair = {
            'left': sample1[label_col],
            'right': sample2[label_col],
            'Y': 0
        }
        pair.update({f'X{j+1}': val for j, val in enumerate(combined_features)})
        pairs.append(pair)
    
    return pd.DataFrame(pairs).sample(frac=1).reset_index(drop=True)

def add_support(data_path, support_dict, test_data, test_ratio=0.1):
    files = glob.glob(f"db/raw/{data_path}/*.csv")
    
    for file_path in files:
        protein_name = file_path.split('/')[-1].split('\\')[-1].split('.')[0]
        df = pd.read_csv(file_path)
            
        target_col = 'potency' if 'potency' in df.columns else 'Y'
        if target_col not in df.columns:
            continue
            
        active = df[df[target_col] == 1].copy()            
        active['fn'] = protein_name
        active['class'] = protein_name
        
        # test/support 분할
        n_test = max(1, int(len(active) * test_ratio))
        test_samples = active.sample(n_test)
        support_samples = active.drop(test_samples.index)
        
        if len(support_samples) > 0:
            support_dict[protein_name] = support_samples
        if len(test_samples) > 0:
            test_data = pd.concat([test_data, test_samples], ignore_index=True)
    
    return support_dict, test_data