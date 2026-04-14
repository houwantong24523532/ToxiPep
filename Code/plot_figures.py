# plot_figures.py - 12张散点图 + 6张柱状图
# DeepKsite 项目

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# 导入特征提取
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (read_fasta, extract_aac, extract_dpc, extract_paac,
                       extract_cksaagp, extract_phyc, extract_ban_features)

# ============ 小红书科研配色 ============
SCATTER_POS = '#E8919A'   # 粉色 - 正样本
SCATTER_NEG = '#7EB6D8'   # 蓝色 - 负样本

BAR_COLORS = {
    'AAC':     '#FF8FAB',
    'DPC':     '#74C0FC',
    'PAAC':    '#B197FC',
    'CKSAAGP': '#63E6BE',
    'PHYC':    '#FFC078',
    'DeepKsite': '#FF6B6B',
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


# ============ 降维 ============
def reduce_tsne(features, labels, max_samples=2000, random_state=42):
    """t-SNE降维，自动采样避免过慢"""
    if len(features) > max_samples:
        np.random.seed(random_state)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_per_class = max_samples // 2
        pos_sample = np.random.choice(pos_idx, min(n_per_class, len(pos_idx)), replace=False)
        neg_sample = np.random.choice(neg_idx, min(n_per_class, len(neg_idx)), replace=False)
        idx = np.concatenate([pos_sample, neg_sample])
        features = features[idx]
        labels = labels[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
    X2d = tsne.fit_transform(X)
    return X2d, labels


# ============ 散点图 ============
def plot_scatter(X2d, labels, title, save_path):
    """画单张散点图"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_style("whitegrid")

    pos_mask = labels == 1
    neg_mask = labels == 0

    ax.scatter(X2d[neg_mask, 0], X2d[neg_mask, 1],
               c=SCATTER_NEG, alpha=0.55, s=20, edgecolors='white',
               linewidths=0.3, label='Negative', zorder=2)
    ax.scatter(X2d[pos_mask, 0], X2d[pos_mask, 1],
               c=SCATTER_POS, alpha=0.55, s=20, edgecolors='white',
               linewidths=0.3, label='Positive', zorder=3)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {save_path}")


# ============ 评估指标 ============
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
    return acc, sn, sp, mcc


# ============ 柱状图 ============
def plot_bar(results, clf_name, save_path):
    """画单张柱状图"""
    sns.set_style("whitegrid")
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'MCC']
    feature_names = list(results.keys())
    colors = [BAR_COLORS[f] for f in feature_names]

    x = np.arange(len(metrics))
    n = len(feature_names)
    width = 0.12
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (fname, color) in enumerate(zip(feature_names, colors)):
        vals = results[fname]
        ax.bar(x + offsets[i], vals, width, label=fname,
               color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_title(clf_name, fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {save_path}")


# ============ 主流程 ============
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, '..', 'Dataset')
    output_dir = os.path.join(base_dir, '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'scatter'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'bar'), exist_ok=True)

    train_pos = os.path.join(dataset_dir, 'train_pos.fasta')
    train_neg = os.path.join(dataset_dir, 'train_neg.fasta')
    test_pos  = os.path.join(dataset_dir, 'test_pos.fasta')
    test_neg  = os.path.join(dataset_dir, 'test_neg.fasta')
    model_path = os.path.join(base_dir, 'best_model.pth')

    # ===== GPU检测 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU已启用: {gpu_name}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  未检测到GPU，使用CPU")
    print("=" * 50)

    # ===== 1. 读取序列 =====
    print("\n[1/5] 读取序列...")
    train_seqs, train_labels = read_fasta(train_pos, train_neg)
    test_seqs,  test_labels  = read_fasta(test_pos,  test_neg)
    print(f"  训练集: {len(train_seqs)} 条 (正:{sum(train_labels==1)} 负:{sum(train_labels==0)})")
    print(f"  测试集: {len(test_seqs)} 条 (正:{sum(test_labels==1)} 负:{sum(test_labels==0)})")

    # ===== 2. 提取手工特征 =====
    print("\n[2/5] 提取手工特征...")
    extractors = {
        'AAC':     extract_aac,
        'DPC':     extract_dpc,
        'PAAC':    extract_paac,
        'CKSAAGP': extract_cksaagp,
        'PHYC':    extract_phyc,
    }

    train_features = {}
    test_features  = {}
    for name, func in extractors.items():
        print(f"\n  --- {name} ---")
        print(f"  训练集:")
        train_features[name] = func(train_seqs)
        print(f"  测试集:")
        test_features[name]  = func(test_seqs)
        print(f"  维度: {train_features[name].shape[1]}")

    # ===== 3. 提取BAN特征 =====
    print("\n[3/5] 提取DeepKsite BAN特征...")
    print("  训练集BAN:")
    train_ban, train_ban_labels = extract_ban_features(train_pos, train_neg, model_path, device)
    print("  测试集BAN:")
    test_ban, test_ban_labels = extract_ban_features(test_pos, test_neg, model_path, device)
    train_features['DeepKsite'] = train_ban
    test_features['DeepKsite']  = test_ban

    # ===== 4. 散点图（12张）=====
    print("\n[4/5] 生成散点图（共12张）...")
    scatter_dir = os.path.join(output_dir, 'scatter')
    all_feat_names = ['AAC', 'DPC', 'PAAC', 'CKSAAGP', 'PHYC', 'DeepKsite']

    for feat_name in all_feat_names:
        # 测试集
        print(f"\n  [{feat_name}] 测试集 t-SNE降维...")
        if feat_name == 'DeepKsite':
            X2d, lbls = reduce_tsne(test_ban, test_ban_labels)
        else:
            X2d, lbls = reduce_tsne(test_features[feat_name], test_labels)
        plot_scatter(X2d, lbls,
                     title=f'{feat_name} - Test Set',
                     save_path=os.path.join(scatter_dir, f'scatter_test_{feat_name}.png'))

        # 训练集
        print(f"  [{feat_name}] 训练集 t-SNE降维...")
        if feat_name == 'DeepKsite':
            X2d, lbls = reduce_tsne(train_ban, train_ban_labels)
        else:
            X2d, lbls = reduce_tsne(train_features[feat_name], train_labels)
        plot_scatter(X2d, lbls,
                     title=f'{feat_name} - Train Set',
                     save_path=os.path.join(scatter_dir, f'scatter_train_{feat_name}.png'))

    # ===== 5. 柱状图（6张）=====
    print("\n[5/5] 训练分类器 + 生成柱状图（共6张）...")
    bar_dir = os.path.join(output_dir, 'bar')

    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('KNN',                 KNeighborsClassifier(n_neighbors=5)),
        ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('SVM',                 SVC(kernel='rbf', random_state=42)),
        ('Naive Bayes',         GaussianNB()),
        ('Gradient Boosting',   GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    # 对训练集采样加速SVM等慢分类器
    MAX_TRAIN = 8000
    for clf_name, clf_obj in classifiers:
        print(f"\n  --- {clf_name} ---")
        results = {}
        for feat_name in all_feat_names:
            X_train = train_features[feat_name]
            X_test  = test_features[feat_name]
            y_train = train_ban_labels if feat_name == 'DeepKsite' else train_labels
            y_test  = test_ban_labels  if feat_name == 'DeepKsite' else test_labels

            # 采样加速
            if len(X_train) > MAX_TRAIN:
                np.random.seed(42)
                idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
                X_train_s = X_train[idx]
                y_train_s = y_train[idx]
            else:
                X_train_s = X_train
                y_train_s = y_train

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_train_s)
            X_te = sc.transform(X_test)

            # 每次需要新实例（避免重复fit问题）
            from sklearn.base import clone
            model = clone(clf_obj)
            model.fit(X_tr, y_train_s)
            y_pred = model.predict(X_te)

            acc, sn, sp, mcc = compute_metrics(y_test, y_pred)
            results[feat_name] = (acc, sn, sp, mcc)
            print(f"    {feat_name}: Acc={acc:.3f} Sn={sn:.3f} Sp={sp:.3f} MCC={mcc:.3f}")

        safe_name = clf_name.replace(' ', '_')
        plot_bar(results, clf_name,
                 save_path=os.path.join(bar_dir, f'bar_{safe_name}.png'))

    print("\n" + "=" * 50)
    print(f"  全部完成！图片保存在: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
