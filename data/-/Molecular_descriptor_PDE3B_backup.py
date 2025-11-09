#필요한 패키지 임포트
from rdkit import Chem
from mordred import Calculator, descriptors
from tqdm import tqdm
import multiprocessing
import pandas as pd

# 2D molecular descriptor 계산 함수 정의
def calc_2d_descriptors(smiles_list, leave_cores=4):
    """
    smiles_list: canonical SMILES 문자열 리스트
    leave_cores: 전체 논리 코어 중 몇 개를 여유용으로 남길지
    반환값: pandas DataFrame (각 행이 분자, 열이 descriptor)
    """
    # 1) 사용할 코어 수 결정
    total_cores = multiprocessing.cpu_count()
    nproc = max(1, total_cores - leave_cores)
    print(f"Using {nproc}/{total_cores} cores for descriptor calculation")

    # 2) SMILES → Mol 객체 변환 (진행바)
    mols = [Chem.MolFromSmiles(smi) for smi in tqdm(smiles_list, desc="Parsing SMILES")]

    # 3) 2D descriptor 계산기 생성
    calc = Calculator(descriptors, ignore_3D=True)

    # 4) pandas DataFrame으로 계산 (병렬 nproc)
    desc_df = calc.pandas(mols, nproc=nproc)

    return desc_df

if __name__ == "__main__":
    # 입력 CSV 로드
    input_path = r"C:\R_PDE3B_data\filtered_canonical_PDE3B_training_data.csv"
    df = pd.read_csv(input_path)
    print("Loaded data shape:", df.shape)

    # SMILES 리스트 추출
    smiles_list = df['canonical_SMILES'].tolist()
    print("총 분자 수:", len(smiles_list))

    # 2D molecular descriptor 계산
    desc_df = calc_2d_descriptors(smiles_list, leave_cores=4)
    print("Descriptor DataFrame shape:", desc_df.shape)
    print(desc_df.head())

    # SMILES 키 컬럼 삽입 및 원본 df와 merge
    desc_df.insert(0, 'canonical_SMILES', smiles_list)
    result = df.merge(desc_df, on='canonical_SMILES', how='left')
    print("Final merged shape:", result.shape)
    print(result.columns.tolist())

    # 결과 CSV로 저장
    output_path = r"C:\R_PDE3B_data\PDE3B_2D_descriptors_0415.csv"
    result.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")