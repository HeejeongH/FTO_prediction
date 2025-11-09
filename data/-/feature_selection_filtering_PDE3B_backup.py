##환경구축
#conda install -c conda-forge scikit-learn
#다시시작

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# 1. 데이터 불러오기
dataset_path = r'C:\R_PDE3B_data\Molecular_descriptors\preprocessed_PDE3B_2D_descriptors.csv'
dataset = pd.read_csv(dataset_path)
print(f"[1] 원본 데이터 shape: {dataset.shape}")

# 2. 앞의 compound 정보 컬럼 (4개) 분리
compound_info = dataset.iloc[:, :4]
descriptor_data = dataset.iloc[:, 4:]
print(f"[1] 원본 descriptor 데이터 shape: {descriptor_data.shape}")

# 3. 문자열이나 이상한 값 포함된 컬럼 제거 함수
def remove_invalid_descriptors(data):
    invalid_cols = []
    for col in data.columns:
        try:
            pd.to_numeric(data[col], errors='raise')  # 숫자로 강제 변환 시도
        except:
            invalid_cols.append(col)
    cleaned_data = data.drop(columns=invalid_cols)
    print(f"[2] 제거된 결측/비정상 값 포함 컬럼 수: {len(invalid_cols)}개")
    return cleaned_data, data[invalid_cols]

descriptor_data_cleaned, descriptors_removed_by_invalid = remove_invalid_descriptors(descriptor_data)
print(f"[3] 결측 제거 후 descriptor shape: {descriptor_data_cleaned.shape}")

# 4. 분산이 낮은 descriptor 제거 함수
def remove_low_variance_descriptors(data, threshold=1e-6):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    selected_columns = data.columns[selector.get_support(indices=True)]
    removed_columns = data.columns.difference(selected_columns)
    selected_data = data[selected_columns]
    removed_data = data[removed_columns]
    print(f"[4] 제거된 낮은 분산 descriptor 수: {len(removed_columns)}개")
    return selected_data, removed_data

# 5. 상관관계 높은 descriptor 제거 함수
def remove_correlated_descriptors(data, threshold=0.9):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_remove = [col for col in upper.columns if any(upper[col] >= threshold)]
    selected_data = data.drop(columns=to_remove)
    removed_data = data[to_remove]
    print(f"[5] 제거된 상관관계 높은 descriptor 수: {len(to_remove)}개")
    return selected_data, removed_data

# 6. 분산이 낮은 descriptor 제거
descriptor_data_after_var, descriptors_removed_by_var = remove_low_variance_descriptors(descriptor_data_cleaned)
print(f"[6] 분산 제거 후 descriptor shape: {descriptor_data_after_var.shape}")

# 7. 상관관계 높은 descriptor 제거
descriptor_data_final, descriptors_removed_by_corr = remove_correlated_descriptors(descriptor_data_after_var)
print(f"[7] 상관관계 제거 후 descriptor shape: {descriptor_data_final.shape}")

# 8. 최종 descriptor + compound 정보 결합
final_data = pd.concat([compound_info, descriptor_data_final], axis=1)
print(f"[8] 최종 데이터 shape (compound info 포함): {final_data.shape}")

# 9. CSV 저장
output_cleaned_path = r'C:\R_PDE3B_data\Molecular_descriptors\filtered_PDE3B_2D_descriptors.csv'
final_data.to_csv(output_cleaned_path, index=False)
print(f"[9] 최종 descriptor 데이터 저장 완료:\n{output_cleaned_path}")

# 10. 제거된 descriptor들 따로 저장 (compound_info 포함)
removed_invalid_with_info = pd.concat([compound_info, descriptors_removed_by_invalid], axis=1)
removed_var_with_info = pd.concat([compound_info, descriptors_removed_by_var], axis=1)
removed_corr_with_info = pd.concat([compound_info, descriptors_removed_by_corr], axis=1)

removed_invalid_with_info.to_csv(r'C:\R_PDE3B_data\Molecular_descriptors\removed_descriptors_by_invalid.csv', index=False)
removed_var_with_info.to_csv(r'C:\R_PDE3B_data\Molecular_descriptors\removed_descriptors_by_variance.csv', index=False)
removed_corr_with_info.to_csv(r'C:\R_PDE3B_data\Molecular_descriptors\removed_descriptors_by_correlation.csv', index=False)

print("[10] 제거된 descriptor 목록 저장 완료 (compound 정보 포함).")