import requests
import pandas as pd
import numpy as np
import glob
import os

from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors

# ChemBL Target Collector Class
class ChemBLTargetCollector:
    def __init__(self):
        self.target = new_client.target
        self.activity = new_client.activity

    def get_target_data(self, target_list, standard_type='IC50', name_map=False):
        all_targets = self._filter_targets(target_list)
        results = self._collect_activities(all_targets, standard_type)
        df = pd.DataFrame(results)
        if name_map == True:
            df['Standard_Name'] = df['target_pref_name'].map(name_map)
        else:
            uniprot_mapping = get_uniprot_mapping_only()
            df['Standard_Name'] = df['target_pref_name'].apply(lambda x: map_with_uniprot_only(x, uniprot_mapping))
        return df

    def _filter_targets(self, target_list):
        all_targets = {}
        for target_i in target_list:
            filtered_target = self.target.filter(target_synonym__icontains=target_i, organism='Homo sapiens')
            for t in filtered_target:
                all_targets[t['target_chembl_id']] = t
        print(len(all_targets))
        return all_targets

    def _collect_activities(self, all_targets, standard_type):
        results = []
        for target_id, t in all_targets.items():
            valid_data = self.activity.filter(
                target_chembl_id=target_id,
                standard_type=standard_type
            )
            for act in valid_data:
                if act['standard_value']:
                    results.append({
                        'target_pref_name': act['target_pref_name'],
                        'target_chembl_id': act['target_chembl_id'],
                        'molecule_pref_name': act['molecule_pref_name'],
                        'molecule_chembl_id': act['molecule_chembl_id'],
                        'canonical_smiles': act['canonical_smiles'],
                        'standard_type': act['standard_type'],
                        'standard_value': act['standard_value'],
                        'standard_units': act['standard_units'],
                        'standard_relation': act['standard_relation']
                    })
        return results

# Uniprot API to get gene symbols and protein names
def get_uniprot_mapping_only():
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        'query': 'organism_id:9606 AND (family:"phosphodiesterase" OR gene:"PDE*" OR gene:"ENPP*" OR gene:"TDP*" OR gene:"PLC*" OR gene:"SMPD*")',
        'format': 'json',
        'fields': 'accession,gene_names,protein_name',
        'size': 500
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    name_to_symbol = {}
    
    for entry in data['results']:
        gene_symbol = ""
        if 'genes' in entry and entry['genes']:
            gene_symbol = entry['genes'][0].get('geneName', {}).get('value', '')
        
        if gene_symbol:
            if 'proteinDescription' in entry:
                rec_name = entry['proteinDescription'].get('recommendedName', {})
                if rec_name:
                    full_name = rec_name.get('fullName', {}).get('value', '')
                    if full_name:
                        name_to_symbol[full_name] = gene_symbol
                
                alt_names = entry['proteinDescription'].get('alternativeNames', [])
                for alt in alt_names:
                    alt_name = alt.get('fullName', {}).get('value', '')
                    if alt_name:
                        name_to_symbol[alt_name] = gene_symbol
    
    return name_to_symbol

def map_with_uniprot_only(target_name, mapping_dict):
    if target_name in mapping_dict:
        return mapping_dict[target_name]
    
    return target_name

# BindingDB
def bindingdb_data(folder):
    tsv_files = glob.glob(f'saved_data/BindingDB/{folder}/*.tsv')
    BDB_df = pd.concat([pd.read_csv(file, sep='\t', low_memory=False) for file in tsv_files], ignore_index=True)
    return BDB_df

def BindingDB_df(folder_name, name_map=False):
  df = bindingdb_data(folder_name)
  df['canonical SMILES'] = df['Ligand SMILES'].apply(to_canonical_smiles)

  IC50_df = df[['Target Source Organism According to Curator or DataSource', 
                'Target Name', 'BindingDB Ligand Name', 'canonical SMILES', 'IC50 (nM)']]\
              .rename(columns={'Target Source Organism According to Curator or DataSource': 'organism'})
  IC50_df_HS = IC50_df[IC50_df['organism'] == 'Homo sapiens'].copy().drop('organism',axis=1)

  if name_map == True:
    IC50_df_HS['Standard_Name'] = IC50_df_HS['Target Name'].map(name_map)
  else:
    uniprot_mapping = get_uniprot_mapping_only()
    IC50_df_HS['Standard_Name'] = IC50_df_HS['Target Name'].apply(lambda x: map_with_uniprot_only(x, uniprot_mapping))

  IC50_df_HS = IC50_df_HS[['Standard_Name', 'canonical SMILES', 'IC50 (nM)']]
  IC50_df_HS = IC50_df_HS.rename(columns = {'canonical SMILES':'canonical_smiles'})
  return IC50_df_HS

# Function to convert SMILES to canonical SMILES
def to_canonical_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def unique_smiles_count(df, target_protein):
    if isinstance(target_protein, list):
        target_smiles = set()
        for protein in target_protein:
            target_smiles.update(df[df['Standard_Name'] == protein]['canonical_smiles'].dropna())
        target_protein_name = '+'.join(target_protein)
    else:
        target_smiles = set(df[df['Standard_Name'] == target_protein]['canonical_smiles'].dropna())
        target_protein_name = target_protein

    if isinstance(target_protein, list):
        other_proteins = df[~df['Standard_Name'].isin(target_protein)]['Standard_Name'].unique()
    else:
        other_proteins = df[df['Standard_Name'] != target_protein]['Standard_Name'].unique()

    result = []
    for protein in other_proteins:
        protein_smiles = set(df[df['Standard_Name'] == protein]['canonical_smiles'].dropna())
        target_unique_from_protein = len(target_smiles - protein_smiles)
        protein_unique_from_target = len(protein_smiles - target_smiles)
        result.append({'Other protein': protein, 
                    'Other protein SMILES count': len(protein_smiles),
                    'Target protein SMILES count': len(target_smiles),
                    'Unique Target protein SMILES': target_unique_from_protein, 
                    'Unique Other protein SMILES': protein_unique_from_target})

    result_df = pd.DataFrame(result).sort_values('Unique Target protein SMILES', ascending=False)
    return result_df

def collect_data(chembl_search_list, binding_db_folder):
    # ChemBL 데이터 수집
    print("Collecting ChemBL data...")
    collector = ChemBLTargetCollector()
    chembl_df = collector.get_target_data(chembl_search_list).dropna()
    print(f"ChemBL data: {len(chembl_df)}")

    # BindingDB 데이터 수집 (# https://www.bindingdb.org/rwd/bind/as.jsp -> "Target Name"에 원하는 이름 검색 후 다운 필요)
    print("Collecting BindingDB data...")
    BDB_df = BindingDB_df(binding_db_folder).dropna()
    print(f"BindingDB data: {len(BDB_df)}")

    # 결과를 엑셀 파일로 저장
    with pd.ExcelWriter(f'saved_data/IC50/{chembl_search_list[0]}_summary.xlsx', mode='w', engine='openpyxl') as writer:
        chembl_df.to_excel(writer, sheet_name='ChemBL', index=False)
        BDB_df.to_excel(writer, sheet_name='BindingDB', index=False)
    
    return chembl_df, BDB_df

# SMILES to FP
def smiles_to_fingerprint(smiles, fp_size=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fp_size)
    
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
    
    arr = np.zeros(fp_size)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Convert DataFrame
def prepare_protein_data(chembl_df, BDB_df, protein_name, output_folder, pos_threshold=10000, neg_threshold=20000):
    os.makedirs(output_folder, exist_ok=True)
    all_compounds = []

    # ChemBL 데이터 처리
    for _, row in chembl_df.iterrows():
        if pd.notna(row['canonical_smiles']):
            ic50_value = row.get('standard_value', None)
            relation = row.get('standard_relation', '')
            
            if pd.notna(ic50_value):
                try:
                    ic50_num = float(ic50_value)
                    
                    if ic50_num <= pos_threshold:
                        if relation in ['=', '<']:
                            potency = 1 
                        else:
                            potency = '-' 
                    elif ic50_num >= neg_threshold:
                        if relation in ['=', '>']:
                            potency = 0 
                        else:
                            potency = '-' 
                    else:
                        potency = '-'
                except (ValueError, TypeError):
                    potency = '-'
            else:
                potency = '-'

            all_compounds.append({
                'smiles': row['canonical_smiles'],
                'potency': potency,
                'source': 'ChemBL'
            })

    # BindingDB 데이터 처리
    for _, row in BDB_df.iterrows():
        if pd.notna(row['canonical_smiles']):
            ic50_value = row.get('IC50 (nM)', None)
            
            if pd.notna(ic50_value):
                try:
                    ic50_num = float(ic50_value)
                    if ic50_num <= pos_threshold:
                        potency = 1  
                    elif ic50_num >= neg_threshold:
                        potency = 0
                    else:
                        potency = '-'
                except (ValueError, TypeError):
                    potency = '-'
            else:
                potency = '-'
            
            all_compounds.append({
                'smiles': row['canonical_smiles'],
                'potency': potency,
                'source': 'BindingDB'
            })

    df = pd.DataFrame(all_compounds)
    
    # 2. 활성/비활성 화합물 필터링
    if "FewshotSet" in output_folder:
        # Few-shot: 활성 화합물만
        filtered_data = df[df['potency'] == 1].copy()
        print(f"처리 중: {protein_name} - {len(filtered_data)} 활성 화합물 (Few-shot)")
    else:
        # Transfer: 활성 + 비활성 화합물
        active_data = df[df['potency'] == 1].copy()
        inactive_data = df[df['potency'] == 0].copy()
        filtered_data = pd.concat([active_data, inactive_data]).reset_index(drop=True)
        print(f"처리 중: {protein_name} - 활성: {len(active_data)}, 비활성: {len(inactive_data)} (Transfer)")
    
    if len(filtered_data) == 0:
        print(f"오류: {protein_name} - 유효한 화합물 없음")
        return
    
    # 3. Fingerprint 계산
    fingerprints = []
    valid_indices = []
    
    for idx, smiles in enumerate(filtered_data['smiles']):
        fp = smiles_to_fingerprint(smiles)
        if fp.sum() > 0:
            fingerprints.append(fp)
            valid_indices.append(idx)
    
    if len(valid_indices) == 0:
        print(f"오류: {protein_name} - 유효한 fingerprint 없음")
        return

    valid_data = filtered_data.iloc[valid_indices].reset_index(drop=True)
    fingerprints = np.array(fingerprints)
    
    # 4. 기존 코드 형식에 맞춰 데이터프레임 생성
    if "FewshotSet" in output_folder:
        # Few-shot: 모든 화합물이 활성
        meta_data = pd.DataFrame({
            'SMILES': valid_data['smiles'],
            'Y': 1,
            'potency': 1
        })
    else:
        # Transfer: 원래 라벨 유지
        meta_data = pd.DataFrame({
            'SMILES': valid_data['smiles'],
            'Y': valid_data['potency'],
            'potency': valid_data['potency']
        })
    
    fp_columns = [f'X{i+1}' for i in range(1024)]
    fp_data = pd.DataFrame(fingerprints, columns=fp_columns)

    result_df = pd.concat([meta_data, fp_data], axis=1)
    
    # 5. CSV 파일로 저장
    output_file = os.path.join(output_folder, f"{protein_name}.csv")
    result_df.to_csv(output_file, index=False)
    print(f"저장 완료: {output_file} - {len(result_df)} 화합물")
    
    return result_df

def convert_protein_data(chembl_df, BDB_df, protein_name, dataset_type="fewshot", pos_threshold=10000, neg_threshold=20000):
    if dataset_type.lower() == "fewshot":
        output_folder = "raw/FewshotSet"
        print(f"{protein_name} 데이터를 Few-shot learning용으로 변환 중... (IC50 임계값: pos =< {pos_threshold}nM / neg => {neg_threshold}nM)")
    elif dataset_type.lower() == "transfer":
        output_folder = "raw/TransferSet"
        print(f"{protein_name} 데이터를 Transfer learning용으로 변환 중... (IC50 임계값: pos =< {pos_threshold}nM / neg => {neg_threshold}nM)")
    else:
        print("오류: dataset_type은 'fewshot' 또는 'transfer'여야 합니다.")
        return None
    
    result_df = prepare_protein_data(chembl_df, BDB_df, protein_name, output_folder, pos_threshold=10000, neg_threshold=20000)
    
    if result_df is not None:
        print(f"변환 완료! {output_folder}/{protein_name}.csv")
        return result_df
    else:
        print("변환 실패!")
        return None