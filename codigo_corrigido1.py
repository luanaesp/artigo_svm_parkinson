# -*- coding: utf-8 -*-
"""
PD Speech Classification – LOSO + SMOTE + Multi-view com Visualizações Completas
"""

import warnings
import time
import numpy as np
import pandas as pd

# Configurar backend não interativo para matplotlib
import matplotlib
matplotlib.use('Agg')  # Backend não interativo para evitar problemas de thread
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve, 
                             precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed, Memory

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURAÇÕES OTIMIZADAS
# ==============================
INPUT_PATH = r"C:\Users\luesp\Downloads\parkinson disease article\UNIFIED_RECONSTRUCTED.csv"
RANDOM_STATE = 42
BALANCE_MODE = "smote"
SMOTE_RATIO = 1.0
MAX_SMOTE_K = 5
USE_STACKING_PRO = False
VERBOSE_EVERY = 10
LIMIT_SUBJECTS = None
N_JOBS = -1

# Configurações de visualização
PLOT_TSNE = True
PLOT_PCA = True
PLOT_LDA = True
PLOT_FEATURE_IMPORTANCE = True
PLOT_CONFUSION_MATRIX = True
PLOT_ROC_CURVES = True

# Configurações de otimização
USE_ORIGINAL_VIEW = True
USE_PCA_VIEW = True
USE_LDA_VIEW = False

TARGET_CANDIDATES = ["label","class","target","y","pd","diagnosis"]
GROUP_CANDIDATES = ["subject_id","subject","id","speaker","patient_id","name","recording","filename"]

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ==============================
# UTILITÁRIOS
# ==============================
def choose_column(cands, cols):
    lower = {c.lower(): c for c in cols}
    for k in cands:
        if k in cols:
            return k
        if k.lower() in lower:
            return lower[k.lower()]
    return None

def safe_smote_fit_resample(X, y):
    """SMOTE com k vizinhos seguro e fallback quando a minoria é muito pequena."""
    cnt0, cnt1 = np.sum(y==0), np.sum(y==1)
    n_min = min(cnt0, cnt1)
    if n_min < 2:
        return X, y
    k = max(1, min(MAX_SMOTE_K, n_min-1))
    maj = 1 if cnt1 > cnt0 else 0
    min_cl = 1 - maj
    n_maj = max(cnt0, cnt1)
    target_min = int(SMOTE_RATIO * n_maj)
    sampling_strategy = {min_cl: target_min}
    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)

def probs_softvote(list_of_prob_arrays, weights=None):
    P = np.vstack(list_of_prob_arrays)
    if weights is None:
        return P.mean(axis=0)
    w = np.array(weights).reshape(-1,1)
    return (P*w).sum(axis=0)/np.sum(w)

def evaluate(y_true, y_prob, threshold=0.5, label="Model"):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = np.nan
    rpt = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    rec0 = rpt.get('0',{}).get('recall', np.nan)
    rec1 = rpt.get('1',{}).get('recall', np.nan)
    print(f"\n[{label}] Acc={acc:.4f} | F1(w)={f1w:.4f} | ROC AUC={roc:.4f} | Recall0={rec0:.4f} | Recall1={rec1:.4f}")
    return dict(model=label, accuracy=acc, f1=f1w, roc_auc=roc, recall_0=rec0, recall_1=rec1)

# ==============================
# FUNÇÕES DE VISUALIZAÇÃO
# ==============================
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(title)
    plt.savefig(f"confusion_matrix_{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Avg Precision = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(f"precision_recall_curve_{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(X, y, title="t-SNE Visualization", n_components=2, perplexity=30):
    if X.shape[0] > 1000:  # Limitar para não sobrecarregar a memória
        indices = np.random.choice(X.shape[0], 1000, replace=False)
        X_tsne = X[indices]
        y_tsne = y[indices]
    else:
        X_tsne = X
        y_tsne = y
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=RANDOM_STATE)
    X_embedded = tsne.fit_transform(X_tsne)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(f"tsne_{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca(X, y, title="PCA Visualization", n_components=2):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"{title} (Variance: {np.sum(pca.explained_variance_ratio_):.4f})")
    plt.savefig(f"pca_{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot de variância explicada
    plt.figure(figsize=(10, 6))
    pca_full = PCA().fit(X)
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.savefig("pca_explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca

def plot_lda(X, y, title="LDA Visualization"):
    if len(np.unique(y)) < 2:
        print("LDA requer pelo menos 2 classes")
        return None
    
    lda = LDA(n_components=min(2, len(np.unique(y))-1))
    X_lda = lda.fit_transform(X, y)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_lda[:, 0], X_lda, c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(f"lda_{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return lda

def plot_feature_importance(feature_importances, feature_names, top_n=20, title="Feature Importance"):
    indices = np.argsort(feature_importances)[::-1]
    
    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.barh(range(top_n), feature_importances[indices][:top_n][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n][::-1]])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f"feature_importance_{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================
# CARREGAMENTO E LIMPEZA INICIAL
# ==============================
def load_data():
    print(f"Lendo: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Shape bruto: {df.shape}")
    # Remove linhas onde o rótulo (label) é NaN
    df.dropna(subset=["label"], inplace=True)

    y_col = choose_column(TARGET_CANDIDATES, df.columns)
    g_col = choose_column(GROUP_CANDIDATES, df.columns)
    if y_col is None or g_col is None:
        raise ValueError(f"Colunas não encontradas — alvo: {y_col} | grupo: {g_col}")

    df[g_col] = df[g_col].astype(str)

    # Lista de colunas de metadados e rótulos disfarçados a serem explicitamente removidas
    # Inclui as colunas identificadas na análise de feature importance que são o próprio rótulo
    meta_and_label_cols_to_drop = [
        'dataset_source', 'recording_id', 'ID__replicated',
        'label_original', 'class_pd_speech', 'Status_replicated', 'ID__replicated',
        'f1_uci2013_train', 'f28_uci2013_train', 'f29_uci2013_train', 'f1_uci2013_test', 'f28_uci2013_test', 'f29_uci2013_test' # Colunas de rótulo e identificadores que causam vazamento de dados
    ]

    non_feats = {y_col, g_col}.union(meta_and_label_cols_to_drop)

    # Tenta converter colunas object para numérico, mas não as inclui como features
    for c in df.columns:
        if c in non_feats:
            continue
        if df[c].dtype == 'object':
            cand = pd.to_numeric(df[c], errors='coerce')
            if cand.notna().mean() > 0.8:
                df[c] = cand
            else: # Se não for majoritariamente numérica, adiciona à lista de remoção
                non_feats.add(c)

    # Remove colunas que são 100% NaN
    all_nan_cols = [c for c in df.columns if c not in non_feats and df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        print(f"Removidas colunas 100% NaN: {len(all_nan_cols)}")

    # Define as features como sendo todas as colunas exceto as não-features
    feature_cols = [c for c in df.columns if c not in non_feats]
    df = df[feature_cols + [y_col, g_col]]

    print(f"Shape após limpeza: {df.shape}")
    print(f"Colunas de features utilizadas: {len(feature_cols)}")

    return df, y_col, g_col

# ==============================
# MODELOS BASE (OTIMIZADOS)
# ==============================
def make_base_models(class_weight=None):
    cw = class_weight if (BALANCE_MODE != "smote") else None
    models = {}
    
    lr = LogisticRegression(max_iter=1000, solver="liblinear", penalty="l2",
                           n_jobs=1, class_weight=cw, random_state=RANDOM_STATE)
    models['LR'] = lr
    
    svm = SVC(kernel="rbf", probability=True, gamma="scale", C=1.0,
              class_weight=cw, random_state=RANDOM_STATE)
    models['SVM'] = svm
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                               n_jobs=1, class_weight=cw, random_state=RANDOM_STATE)
    models['RF'] = rf
        
    return models

# ==============================
# PROCESSAMENTO DE FOLD
# ==============================
def process_fold(fold_data):
    i, (tr, te), groups, X_df, y, num_cols, cat_cols, n_subj, t0 = fold_data
    
    subj = groups[te][0]
    if (i == 0) or ((i+1) % VERBOSE_EVERY == 0):
        print(f"[Fold {i+1}/{n_subj}] sujeito teste: {subj} | elapsed {(time.time()-t0)/60:.1f} min")

    X_tr_df, X_te_df = X_df.iloc[tr], X_df.iloc[te]
    y_tr, y_te = y[tr], y[te]

    # Pré-processamento
    low_card = [c for c in cat_cols if X_tr_df[c].nunique() <= 20] if cat_cols else []

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if low_card:
        transformers.append(("cat", cat_pipe, low_card))

    pre = ColumnTransformer(transformers=transformers, remainder='drop')
    X_tr = pre.fit_transform(X_tr_df)
    X_te = pre.transform(X_te_df)

    # Garante dtype compacto e sem NaN
    X_tr = np.asarray(X_tr, dtype=np.float32)
    X_te = np.asarray(X_te, dtype=np.float32)
    
    if np.isnan(X_tr).any() or np.isnan(X_te).any():
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0)

    # Redução de dimensionalidade com seleção de variância
    selector = VarianceThreshold(threshold=0.01)
    X_tr = selector.fit_transform(X_tr)
    X_te = selector.transform(X_te)

    # Lista para armazenar probabilidades de cada visão
    all_probs = []

    # (1) Visão Original
    if USE_ORIGINAL_VIEW:
        X_tr_orig, y_tr_orig = X_tr, y_tr
        if BALANCE_MODE == "smote":
            X_tr_orig, y_tr_orig = safe_smote_fit_resample(X_tr_orig, y_tr_orig)
        
        class_weight = "balanced" if BALANCE_MODE == "weights" else None
        base_models = make_base_models(class_weight=class_weight)
        
        pv = []
        for name, clf in base_models.items():
            clf.fit(X_tr_orig, y_tr_orig)
            pv.append(clf.predict_proba(X_te)[:, 1])
        
        prob_orig = probs_softvote(pv)
        all_probs.append(prob_orig)

    # (2) Visão PCA
    if USE_PCA_VIEW and X_tr.shape[1] > 1:
        n_components_pca = min(X_tr.shape[1], max(1, int(0.95 * X_tr.shape[1])))
        
        pca = PCA(n_components=n_components_pca, svd_solver="randomized", random_state=RANDOM_STATE)
        X_tr_p = pca.fit_transform(X_tr)
        X_te_p = pca.transform(X_te)
        y_tr_p = y_tr.copy()
        
        if BALANCE_MODE == "smote":
            X_tr_p, y_tr_p = safe_smote_fit_resample(X_tr_p, y_tr_p)
        
        class_weight = "balanced" if BALANCE_MODE == "weights" else None
        base_models = make_base_models(class_weight=class_weight)
        
        pv = []
        for name, clf in base_models.items():
            clf.fit(X_tr_p, y_tr_p)
            pv.append(clf.predict_proba(X_te_p)[:, 1])
        
        prob_pca = probs_softvote(pv)
        all_probs.append(prob_pca)

    # (3) Visão LDA
    if USE_LDA_VIEW and len(np.unique(y_tr)) >= 2 and X_tr.shape[1] > 1:
        lda = LDA(solver="eigen", shrinkage="auto")
        X_tr_l = lda.fit_transform(X_tr, y_tr)
        X_te_l = lda.transform(X_te)
        y_tr_l = y_tr.copy()
        
        if BALANCE_MODE == "smote":
            X_tr_l, y_tr_l = safe_smote_fit_resample(X_tr_l, y_tr_l)
        
        class_weight = "balanced" if BALANCE_MODE == "weights" else None
        base_models = make_base_models(class_weight=class_weight)
        
        pv = []
        for name, clf in base_models.items():
            clf.fit(X_tr_l, y_tr_l)
            pv.append(clf.predict_proba(X_te_l)[:, 1])
        
        prob_lda = probs_softvote(pv)
        all_probs.append(prob_lda)

    # Ensemble entre visões
    if len(all_probs) > 0:
        p_final = probs_softvote(all_probs)
    else:
        # Fallback se nenhuma visão estiver ativa
        p_final = np.zeros(len(y_te))

    return y_te.tolist(), p_final.tolist()

# ==============================
# LOSO PARALELIZADO
# ==============================
def run_loso_parallel(df, y_col, g_col):
    cols = [c for c in df.columns if c not in [y_col, g_col]]
    y = df[y_col].astype(int).values
    groups = df[g_col].astype(str).values
    X_df = df[cols].copy()

    if LIMIT_SUBJECTS:
        keep = []
        for s in pd.unique(groups):
            keep.append(s)
            if len(keep) >= LIMIT_SUBJECTS: 
                break
        mask = np.isin(groups, keep)
        X_df, y, groups = X_df.loc[mask], y[mask], groups[mask]

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    logo = LeaveOneGroupOut()
    n_subj = len(np.unique(groups))
    
    print(f"Iniciando LOSO com {n_subj} sujeitos...")
    print(f"Configuração: Original={USE_ORIGINAL_VIEW}, PCA={USE_PCA_VIEW}, LDA={USE_LDA_VIEW}")
    t0 = time.time()

    # Preparar dados para processamento paralelo
    fold_data = []
    for i, (tr, te) in enumerate(logo.split(X_df, y, groups)):
        fold_data.append((i, (tr, te), groups, X_df, y, num_cols, cat_cols, n_subj, t0))

    # Processar em paralelo
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_fold)(data) for data in fold_data
    )

    # Coletar resultados
    y_true_all, proba_all = [], []
    for y_te, p_final in results:
        y_true_all.extend(y_te)
        proba_all.extend(p_final)

    y_true_all = np.array(y_true_all)
    proba_all = np.array(proba_all)
    
    _ = evaluate(y_true_all, proba_all, threshold=0.5, label="Ensemble (soft-vote)")
    return dict(y_true=y_true_all, y_prob=proba_all)

# ==============================
# ANÁLISE DE CARACTERÍSTICAS IMPORTANTES
# ==============================
def analyze_feature_importance(df, y_col, g_col):
    print("\nAnalisando importância das características...")
    
    cols = [c for c in df.columns if c not in [y_col, g_col]]
    y = df[y_col].astype(int).values
    X_df = df[cols].copy()

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    # Pré-processamento simplificado para a análise de importância
    pre = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ], remainder='drop'
    )

    X_processed = pre.fit_transform(X_df)
    
    # Obter nomes das features após o OneHotEncoding
    try:
        ohe_feature_names = pre.named_transformers_['cat'].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(ohe_feature_names)
    except (AttributeError, KeyError):
        feature_names = num_cols

    # Treinar Random Forest para obter a importância
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_processed, y)

    if PLOT_FEATURE_IMPORTANCE:
        plot_feature_importance(rf.feature_importances_, feature_names, top_n=30)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    df, y_col, g_col = load_data()
    
    # Análise e visualização exploratória
    print("\nEstatísticas do dataset:")
    print(df[y_col].value_counts())
    
    # Preparar dados para visualização (imputação e escala)
    X_vis = df.drop(columns=[y_col, g_col])
    y_vis = df[y_col].values
    
    vis_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    X_vis_processed = vis_pipe.fit_transform(X_vis)

    if PLOT_TSNE:
        plot_tsne(X_vis_processed, y_vis, title="t-SNE do Dataset Completo")
    if PLOT_PCA:
        plot_pca(X_vis_processed, y_vis, title="PCA do Dataset Completo")
    if PLOT_LDA:
        plot_lda(X_vis_processed, y_vis, title="LDA do Dataset Completo")

    # Executar validação cruzada
    results = run_loso_parallel(df, y_col, g_col)
    
    # Visualizar resultados agregados
    if PLOT_CONFUSION_MATRIX:
        y_pred_all = (results['y_prob'] >= 0.5).astype(int)
        plot_confusion_matrix(results['y_true'], y_pred_all, title="Matriz de Confusão – LOSO")
    
    if PLOT_ROC_CURVES:
        plot_roc_curve(results['y_true'], results['y_prob'], title="Curva ROC – LOSO")
        plot_precision_recall_curve(results['y_true'], results['y_prob'], title="Curva Precisão-Recall – LOSO")

    # Análise de características
    analyze_feature_importance(df, y_col, g_col)

    print("\nAnálise concluída.")


