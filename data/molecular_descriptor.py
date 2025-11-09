from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors
from mordred import Calculator, descriptors
from tqdm import tqdm
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import multiprocessing
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# SMILES 리스트를 받아서 Descriptors를 계산하는 함수
def calc_descriptors(smiles_list, leave_cores=4, ignore3D=True):
    mols = []
    valid_indices = []
    
    for i, smi in enumerate(tqdm(smiles_list, desc="Processing SMILES")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Invalid SMILES at index {i}: {smi}")
                continue
                
            mol = Chem.AddHs(mol)
            
            if not ignore3D:
                # 3D 좌표 생성 시도
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result == 0:  # 성공
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                    except:
                        try:
                            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                        except:
                            pass  # 최적화 실패해도 진행
                    
                    mols.append(mol)
                    valid_indices.append(i)
                else:
                    # 3D 생성 실패 - 이 분자는 제외
                    print(f"3D embedding failed for SMILES {i}: {smi}")
                    continue
            else:
                # 2D만 계산
                mols.append(mol)
                valid_indices.append(i)
                
        except Exception as e:
            print(f"Error processing SMILES {i}: {smi}, Error: {e}")
            continue
    
    print(f"Valid molecules for descriptor calculation: {len(mols)}/{len(smiles_list)}")
    
    if len(mols) == 0:
        print("No valid molecules!")
        return pd.DataFrame()
    
    # Mordred Calculator로 descriptor 계산
    calc = Calculator(descriptors, ignore_3D=ignore3D)
    
    total_cores = multiprocessing.cpu_count()
    nproc = max(1, total_cores - leave_cores)
    print(f"Using {nproc}/{total_cores} cores for descriptor calculation")
    
    desc_df = calc.pandas(mols, nproc=nproc)
    
    # 유효한 SMILES만 반환 (길이 맞춤)
    valid_smiles = [smiles_list[i] for i in valid_indices]
    desc_df.insert(0, 'canonical_SMILES', valid_smiles)
    
    return desc_df

# 문자열이나 이상한 값 포함된 컬럼 제거 함수
def remove_invalid_descriptors(data):
    invalid_cols = []
    
    for col in data.columns:
        col_data = data[col].dropna()
        
        if len(col_data) == 0:  # 모든 값이 결측치
            invalid_cols.append(col)
            continue
        
        try:
            numeric_data = pd.to_numeric(col_data, errors='raise')
            
            unique_vals = set(numeric_data.unique())
            if len(unique_vals) <= 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
                invalid_cols.append(col)
                continue
                
        except (ValueError, TypeError):
            invalid_cols.append(col)
            continue
    
    cleaned_data = data.drop(columns=invalid_cols)
    removed_data = data[invalid_cols] if invalid_cols else pd.DataFrame()
    
    print(f"제거된 결측/비정상 값 포함 컬럼 수: {len(invalid_cols)}개")
    return cleaned_data, removed_data

# 분산이 낮은 descriptor 제거
def remove_low_variance_descriptors(data, threshold=1e-6):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    selected_columns = data.columns[selector.get_support(indices=True)]
    removed_columns = data.columns.difference(selected_columns)
    selected_data = data[selected_columns]
    removed_data = data[removed_columns]
    print(f"제거된 낮은 분산 descriptor 수: {len(removed_columns)}개")
    return selected_data, removed_data

# 상관관계 높은 descriptor 제거
def remove_correlated_descriptors(data, threshold=0.9):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_remove = [col for col in upper.columns if any(upper[col] >= threshold)]
    selected_data = data.drop(columns=to_remove)
    removed_data = data[to_remove]
    print(f"제거된 상관관계 높은 descriptor 수: {len(to_remove)}개")
    return selected_data, removed_data

def smiles_to_fingerprint(smiles, fp_size=1024, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fp_size)
    
    try:        
        generator = GetMorganGenerator(radius=radius, fpSize=fp_size)
        fp = generator.GetFingerprint(mol)
        
        arr = np.zeros(fp_size)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
        
    except ImportError:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
        arr = np.zeros(fp_size)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

# 특정 Descriptors만 계산하는 함수
def calc_specific_descriptors(smiles_list, descriptor_list=None, leave_cores=4, ignore3D=True):
    mols = []
    valid_indices = []
    
    for i, smi in enumerate(tqdm(smiles_list, desc="Processing SMILES")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
                
            mol = Chem.AddHs(mol)
            
            if not ignore3D:
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result == 0:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                    except:
                        try:
                            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                        except:
                            pass
                    mols.append(mol)
                    valid_indices.append(i)
                else:
                    continue
            else:
                mols.append(mol)
                valid_indices.append(i)
                
        except Exception as e:
            continue
    
    print(f"Valid molecules: {len(mols)}/{len(smiles_list)}")
    
    if len(mols) == 0:
        return pd.DataFrame()
    
    if descriptor_list is None:
        calc = Calculator(descriptors, ignore_3D=ignore3D)
    else:
        selected_descriptors = []
        all_descriptors = descriptors.all()
        
        for desc_name in descriptor_list:
            for desc in all_descriptors:
                if str(desc) == desc_name or desc.__class__.__name__ == desc_name:
                    selected_descriptors.append(desc)
                    break
        
        if not selected_descriptors:
            print("선택된 descriptor가 없습니다.")
            return pd.DataFrame()
            
        calc = Calculator(selected_descriptors, ignore_3D=ignore3D)
    
    total_cores = multiprocessing.cpu_count()
    nproc = max(1, total_cores - leave_cores)
    print(f"Using {nproc}/{total_cores} cores")
    
    desc_df = calc.pandas(mols, nproc=nproc)
    
    valid_smiles = [smiles_list[i] for i in valid_indices]
    desc_df.insert(0, 'canonical_SMILES', valid_smiles)
    
    return desc_df
