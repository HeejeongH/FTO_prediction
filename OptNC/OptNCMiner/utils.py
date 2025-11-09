import logging
import numpy as np
import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
                             roc_auc_score, average_precision_score, multilabel_confusion_matrix)
from model import OptNCMiner, save_model
import os

class DataManager:
    @staticmethod
    def prepare_data(name, is_transfer=False):
        train_raw = pd.read_csv(f"db/processed/{name}/{name}_sim_train.csv")
        data = pd.read_csv(f"db/processed/{name}/{name}_sim_data.csv")
        
        x_cols = [col for col in data.columns if col.startswith('X')]
        target_col = 'Y' if is_transfer else ('potency' if 'potency' in data.columns else 'Y')
        
        train, valid = train_test_split(data, test_size=0.1)
        
        x_tr = train[x_cols].values.astype(np.float64)
        y_tr = train[target_col].values.astype(np.float64).reshape(-1, 1)
        x_val = valid[x_cols].values.astype(np.float64)
        y_val = valid[target_col].values.astype(np.float64).reshape(-1, 1)
        
        support_set = dict(tuple(train_raw.groupby('fn')))
        input_dim = len(x_cols) // 2
        
        return x_tr, y_tr, x_val, y_val, support_set, input_dim

    @staticmethod
    def save_results(y_matrix, y_proba, results_df, save_name):
        if save_name is not None:
            y_matrix.to_csv(f"result/{save_name}_yproba-matrix.csv", index=False)
            y_proba.to_csv(f"result/{save_name}_yproba.csv", index=False)
            results_df.to_csv(f"result/{save_name}_results.csv")

class Evaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0,
            'ap': average_precision_score(y_true, y_pred),
            'count': y_true.sum()
        }

    @staticmethod
    def evaluation(y_matrix, y_test, thresh=0.5):
        exclude_cols = {'LABEL', 'SMILES', 'class', 'fn'}
        proteins = [col for col in y_matrix.columns if col not in exclude_cols]
        results = []
        
        for protein in proteins:
            y_pred = (y_matrix[protein] >= thresh).astype(int)
            y_true = (y_test == protein).astype(int)
            
            if y_true.sum() > 0:
                metrics = Evaluator.calculate_metrics(y_true, y_pred)
                conf = confusion_matrix(y_true, y_pred).ravel()
                results.append([protein] + list(metrics.values()) + conf.tolist())
        
        y_pred = (y_matrix[proteins].max(axis=1) < thresh).astype(int)
        y_true = (y_test == '-').astype(int)
        metrics = Evaluator.calculate_metrics(y_true, y_pred)
        conf = confusion_matrix(y_true, y_pred).ravel()
        results.append(['-'] + list(metrics.values()) + conf.tolist())
        
        columns = ['protein', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'ap', 'count', 'tn', 'fp', 'fn', 'tp']
        results_df = pd.DataFrame(results, columns=columns)
        
        # Add summary rows
        if len(results_df) > 0:
            total_count = max(results_df['count'].sum(), 1)
            numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
            
            avg_row = ['average'] + [results_df[col].mean() for col in numeric_cols] + [total_count] + [-1]*4
            wavg_row = ['weighted_avg'] + [(results_df[col] * results_df['count']).sum() / total_count for col in numeric_cols] + [total_count] + [-1]*4
            
            results_df = pd.concat([results_df, pd.DataFrame([avg_row, wavg_row], columns=columns)], ignore_index=True)
        
        return results_df

    @staticmethod
    def evaluation_multilabel(y_matrix, y_test, thresh=0.5):
        exclude_cols = {'LABEL', 'SMILES', 'class', 'fn'}
        proteins = [col for col in y_matrix.columns if col not in exclude_cols]
        y_pred = (y_matrix[proteins] >= thresh).astype(int)
        
        valid_proteins = [p for p in proteins if y_test[p].sum() > 0]
        
        return {
            'accuracy': accuracy_score(y_test[proteins], y_pred),
            'precision': precision_score(y_test[proteins], y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test[proteins], y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test[proteins], y_pred, average='weighted', zero_division=0),
            'auc': roc_auc_score(y_test[valid_proteins], y_matrix[valid_proteins], average='weighted') if valid_proteins else 0,
            'ap': average_precision_score(y_test[valid_proteins], y_matrix[valid_proteins], average='weighted') if valid_proteins else 0,
            'confusion_matrix': multilabel_confusion_matrix(y_test[proteins], y_pred)
        }

class Predictor:
    @staticmethod
    def _prepare_support_data(model, n_support, support=None):
        support_set = support if support is not None else model.support_pos
        support_data = pd.DataFrame()
        
        for key, temp in support_set.items():
            target_col = 'potency' if 'potency' in temp.columns else 'Y'
            pos_samples = temp[temp[target_col] == 1]
            sampled = pos_samples.sample(min(n_support, len(pos_samples)))
            support_data = pd.concat([support_data, sampled])
        
        x_cols = [col for col in support_data.columns if col.startswith('X') and len(col) <= 5]
        sup_features = support_data[x_cols].values.astype(np.float64)
        
        return sup_features, support_data['fn'].values

    @staticmethod
    def _batch_predict(x, sup_features, fn_names, iter_size, predict_func):
        results = []
        for i in range(0, len(x), iter_size):
            batch_x = x[i:min(i+iter_size, len(x))]
            
            left = np.tile(sup_features, (len(batch_x), 1))
            right = np.repeat(batch_x, len(sup_features), axis=0)
            
            predictions = predict_func(left, right)
            batch_results = predictions.reshape(len(batch_x), len(sup_features))
            results.append(pd.DataFrame(batch_results, columns=fn_names))
        
        return pd.concat(results, ignore_index=True)

    @staticmethod
    def predict_baseline(model, x, n_support=100, iter_size=100, random_seed=777):
        np.random.seed(random_seed)
        sup_features, fn_names = Predictor._prepare_support_data(model, n_support)

        def cosine_similarity_func(left, right):
            return t.cosine_similarity(t.tensor(left, dtype=t.float32), t.tensor(right, dtype=t.float32)).detach().numpy()
        
        return Predictor._batch_predict(x, sup_features, fn_names, iter_size, cosine_similarity_func)

    @staticmethod
    def predict_fewshot(model, x, n_support=100, iter_size=500, support=None, random_seed=777):
        np.random.seed(random_seed)
        sup_features, fn_names = Predictor._prepare_support_data(model, n_support, support)
        
        def model_predict_func(left, right):
            return model.predict2(left, right)
        
        return Predictor._batch_predict(x, sup_features, fn_names, iter_size, model_predict_func)

class Trainer:
    @staticmethod
    def _test_common(model, name, params, test_raw=None, thresh=0.5, seed=777):
        n_support = params['nsupport']
        n_iter = params['niter']
        
        if test_raw is None:
            test_raw = pd.read_csv(f"db/processed/{name}/{name}_sim_test.csv")
        
        x_cols = [col for col in test_raw.columns if col.startswith('X') and len(col) <= 5]
        x_test = test_raw[x_cols].values.astype(np.float64)
        y_test = test_raw['class']
        
        y_proba = Predictor.predict_fewshot(model, x_test, n_support=n_support, iter_size=n_iter, random_seed=seed)
        
        y_matrix = y_proba.groupby(axis=1, level=0).max()
        y_matrix['LABEL'] = y_test
        
        label_col = 'SMILES' if 'SMILES' in test_raw.columns else 'smiles'
        if label_col in test_raw.columns:
            y_matrix[label_col] = test_raw[label_col]
        
        y_matrix = y_matrix.reset_index(drop=True)
        
        return y_proba, y_matrix, y_test

    @staticmethod
    def train_cycle(params):
        name = params['name']
        head_dims = params['headshape']
        dropout = params['dr']
        lr = params['lr']
        
        model_name = f"{name}_{dropout:.2f}_{'-'.join(map(str, head_dims))}"
        
        x_tr, y_tr, x_val, y_val, support_set, input_dim = DataManager.prepare_data(name)
        
        model = OptNCMiner(input_dim=input_dim, head_dims=head_dims, dropout=dropout)
        losses, optimizer = model.fit(x_tr, y_tr, valset=(x_val, y_val), support=support_set, epochs=500, lr=lr, es_thresh=50)
        
        save_model(model, f"model/model_{model_name}.pt")
        return model, losses, optimizer

    @staticmethod
    def transfer_train_cycle(model, params):
        name = params['name']
        tf_name = params['tfname']
        head_dims = params['headshape']
        dropout = params['dr']
        lr = params['lr']
        
        model_name = f"{name}_{dropout:.2f}_{'-'.join(map(str, head_dims))}_tf{tf_name}"
        
        x_tr, y_tr, x_val, y_val, support_set, _ = DataManager.prepare_data(tf_name, is_transfer=True)
        
        losses, optimizer = model.fit(x_tr, y_tr, valset=(x_val, y_val), support=support_set, epochs=100, lr=lr, es_thresh=10)
        
        save_model(model, f"model/model_{model_name}.pt")
        return model, losses, optimizer

    @staticmethod
    def test_cycle(model, params, save_name=None, test_raw=None, thresh=0.5, seed=777):
        if os.path.exists(f"result/{save_name}_yproba-matrix.csv"):
            print("test_file exist")
            y_matrix = pd.read_csv(f"result/{save_name}_yproba-matrix.csv")
            y_proba = pd.read_csv(f"result/{save_name}_yproba.csv")
            results_df = pd.read_csv(f"result/{save_name}_results.csv")
        else:
            name = params['name']
            y_proba, y_matrix, y_test = Trainer._test_common(model, name, params, test_raw, thresh, seed)
            results_df = Evaluator.evaluation(y_matrix, y_test, thresh=thresh)
            DataManager.save_results(y_matrix, y_proba, results_df, save_name)
        return y_proba, y_matrix

    @staticmethod
    def tf_test_cycle(model, params, save_name=None, test_raw=None, thresh=0.5, seed=777):
        if os.path.exists(f"result/{save_name}_yproba-matrix.csv"):
            print("test_file exist")
            y_matrix = pd.read_csv(f"result/{save_name}_yproba-matrix.csv")
            y_proba = pd.read_csv(f"result/{save_name}_yproba.csv")
            results_df = pd.read_csv(f"result/{save_name}_results.csv")
        else:
            name = params['tfname']
            y_proba, y_matrix, y_test = Trainer._test_common(model, name, params, test_raw, thresh, seed)
            results_df = Evaluator.evaluation(y_matrix, y_test, thresh=thresh)
            DataManager.save_results(y_matrix, y_proba, results_df, save_name)
        return y_proba, y_matrix
    
class NCDiscovery:
    @staticmethod
    def prepare_target_support(target_csv_path):
        df = pd.read_csv(target_csv_path)
        active_compounds = df[df['Y'] == 1]
        
        x_cols = [col for col in df.columns if col.startswith('X')]
        features = active_compounds[x_cols].values.astype(np.float64)
        smiles = active_compounds['SMILES'].values
        
        return features, smiles
    
    @staticmethod
    def predict_target_binding(model, query_features, target_features, batch_size=500):
        model.eval()
        predictions = []
        
        with t.no_grad():
            for i in range(0, len(query_features), batch_size):
                batch_query = query_features[i:i+batch_size]
                
                n_query = len(batch_query)
                n_target = len(target_features)
                
                left = np.tile(target_features, (n_query, 1))
                right = np.repeat(batch_query, n_target, axis=0)
                
                combined = t.cat([
                    t.tensor(left, dtype=t.float32), 
                    t.tensor(right, dtype=t.float32)
                ], dim=1)
                
                batch_pred = model(combined).numpy()
                batch_pred = batch_pred.reshape(n_query, n_target)
                
                max_similarity = np.max(batch_pred, axis=1)
                predictions.extend(max_similarity)
        
        return np.array(predictions)
    
    @staticmethod
    def screen_compounds(model, target_csv_path, query_df, threshold=0.5, top_k=100):
        target_features, _ = NCDiscovery.prepare_target_support(target_csv_path)
        
        x_cols = [col for col in query_df.columns if col.startswith('X')]
        query_features = query_df[x_cols].values.astype(np.float64)
        
        predictions = NCDiscovery.predict_target_binding(model, query_features, target_features)
        results = query_df.copy()
        results['binding_score'] = predictions
        
        candidates = results[results['binding_score'] >= threshold]
        candidates = candidates.sort_values('binding_score', ascending=False)
        
        return candidates.head(top_k)

