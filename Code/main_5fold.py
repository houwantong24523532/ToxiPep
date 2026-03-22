
import sys
from model import *
from dataset import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_model(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, graph_features, labels = batch
        input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, graph_features, device)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(train_loader)


def model_inference(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)
            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            pos_probs = probs[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(pos_probs.cpu().numpy())
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    avg_loss = total_loss / len(test_loader)
    return all_probs, all_labels, avg_loss


def compute_metrics(all_probs, all_labels, threshold=0.5):
    all_preds = (all_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
    except:
        roc_auc = 0
        pr_auc = 0
    return {
        'sn': sensitivity, 'sp': specificity, 'acc': accuracy,
        'mcc': mcc, 'precision': precision, 'f1': f1,
        'roc_auc': roc_auc, 'pr_auc': pr_auc
    }


def find_optimal_threshold(all_probs, all_labels):
    best_threshold = 0.5
    best_mcc = -1
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    return best_threshold, best_mcc


# ============== 方案配置 ==============
SCHEMES = {
    'ban': {
        'name': 'ToxiPep (BAN)',
        'focal_gamma': 1.0,
        'lr': 0.0003,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'patience': 5,
    },
    'without_sequence_encoder': {
        'name': 'Without Sequence Encoder',
        'focal_gamma': 2.0,
        'lr': 0.0001,
        'weight_decay': 0.05,
        'n_epochs': 20,
        'patience': 5,
    },
    'without_structure_encoder': {
        'name': 'Without Structure Encoder',
        'focal_gamma': 3.0,
        'lr': 0.0001,
        'weight_decay': 0.05,
        'n_epochs': 20,
        'patience': 5,
    },
    'cross_attention': {
        'name': 'Cross Attention',
        'focal_gamma': 2.0,
        'lr': 0.0003,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'patience': 5,
    },
    'information_bottleneck': {
        'name': 'Information Bottleneck',
        'focal_gamma': 2.0,
        'lr': 0.0003,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'patience': 5,
    },
    'concatenation': {
        'name': 'Concatenation',
        'focal_gamma': 2.0,
        'lr': 0.0003,
        'weight_decay': 0.01,
        'n_epochs': 20,
        'patience': 5,
    },
}


def create_model(scheme_key, device):
    vocab_size = len(Pep_residue2idx)
    d_model, d_ff, n_layers, n_heads, max_len = 256, 512, 2, 4, 33
    structural_config = {
        "embedding_dim": 21, "max_seq_len": 33, "filter_num": 64,
        "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
    }

    if scheme_key == 'ban':
        return ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                             structural_config=structural_config).to(device)
    elif scheme_key == 'without_sequence_encoder':
        return ToxiPep_WithoutSequenceEncoder(d_model=d_model, structural_config=structural_config).to(device)
    elif scheme_key == 'without_structure_encoder':
        return ToxiPep_WithoutStructureEncoder(vocab_size, d_model, d_ff, n_layers, n_heads, max_len).to(device)
    elif scheme_key == 'cross_attention':
        return ToxiPep_CrossAttention(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                                     structural_config=structural_config).to(device)
    elif scheme_key == 'information_bottleneck':
        return ToxiPep_InformationBottleneck(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                                            structural_config=structural_config, k=128).to(device)
    elif scheme_key == 'concatenation':
        return ToxiPep_Concatenation(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                                    structural_config=structural_config).to(device)


def load_all_data():
    max_len = 33
    print("正在加载训练数据...")
    all_sequences, all_graph_features, all_labels = load_data_from_separate_files(
        '../Dataset/train_pos.fasta', '../Dataset/train_neg.fasta', max_len=max_len + 1)
    return all_sequences, all_graph_features, all_labels


def run_5fold(scheme_key, preloaded_data=None):
    config = SCHEMES[scheme_key]
    scheme_name = config['name']
    n_epochs = config['n_epochs']

    print("\n" + "=" * 70)
    print(f"  五折交叉验证: {scheme_name}")
    print("=" * 70)

    max_len = 33

    if preloaded_data is not None:
        all_sequences, all_graph_features, all_labels = preloaded_data
    else:
        all_sequences, all_graph_features, all_labels = load_all_data()

    all_labels_np = np.array(all_labels)

    print(f"训练数据总量: {len(all_labels)} (正样本: {sum(all_labels)}, 负样本: {len(all_labels) - sum(all_labels)})")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 五折划分
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_labels_np, all_labels_np)):
        fold = fold_idx + 1
        print(f"\n{'─'*60}")
        print(f"  第 {fold}/5 折")
        print(f"{'─'*60}")

        train_seq = [all_sequences[i] for i in train_idx]
        train_gf = [all_graph_features[i] for i in train_idx]
        train_lb = [all_labels[i] for i in train_idx]
        val_seq = [all_sequences[i] for i in val_idx]
        val_gf = [all_graph_features[i] for i in val_idx]
        val_lb = [all_labels[i] for i in val_idx]

        print(f"  训练集: {len(train_lb)} (正: {sum(train_lb)}, 负: {len(train_lb)-sum(train_lb)})")
        print(f"  验证集: {len(val_lb)} (正: {sum(val_lb)}, 负: {len(val_lb)-sum(val_lb)})")

        train_dataset = MyDataSet(train_seq, train_gf, train_lb)
        val_dataset = MyDataSet(val_seq, val_gf, val_lb)
        train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=False)

        # 创建模型
        model = create_model(scheme_key, device)

        # 损失函数
        n_pos = sum(train_lb)
        n_neg = len(train_lb) - n_pos
        pos_weight = np.sqrt(n_neg / n_pos)
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=config['focal_gamma'])

        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_score = -1.0
        best_model_path = f"5fold_{scheme_key}_fold{fold}.pth"
        optimal_threshold = 0.5
        no_improve_count = 0
        early_stop_patience = config['patience']

        print(f"  开始训练，最多 {n_epochs} 轮 (Early Stopping: 连续 {early_stop_patience} 轮无提升则停止)...")
        print(f"  {'Epoch':>5} | {'Loss':>7} | {'Sn':>6} | {'Sp':>6} | {'Acc':>6} | {'MCC':>6} | {'F1':>6} | {'AUC':>6} | {'PR-AUC':>6}")
        print(f"  {'-'*85}")

        for epoch in range(n_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device, scheduler)
            all_probs, all_labels_val, _ = model_inference(model, val_loader, criterion, device)
            opt_thresh, _ = find_optimal_threshold(all_probs, all_labels_val)
            metrics = compute_metrics(all_probs, all_labels_val, threshold=opt_thresh)

            print(f'  {epoch+1:5d} | {train_loss:7.4f} | '
                  f'{metrics["sn"]:6.4f} | {metrics["sp"]:6.4f} | {metrics["acc"]:6.4f} | '
                  f'{metrics["mcc"]:6.4f} | {metrics["f1"]:6.4f} | '
                  f'{metrics["roc_auc"]:6.4f} | {metrics["pr_auc"]:6.4f}', end='')

            composite_score = 0.5 * metrics["mcc"] + 0.5 * metrics["pr_auc"]
            if composite_score > best_score:
                best_score = composite_score
                optimal_threshold = opt_thresh
                no_improve_count = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimal_threshold': optimal_threshold,
                    'metrics': metrics,
                    'fold': fold
                }, best_model_path)
                print(f'  *best*')
            else:
                no_improve_count += 1
                print()
                if no_improve_count >= early_stop_patience:
                    print(f"  >>> Early Stopping: 连续 {early_stop_patience} 轮无提升，在第 {epoch+1} 轮停止")
                    break

        # 加载该折最佳模型评估
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        all_probs, all_labels_val, _ = model_inference(model, val_loader, criterion, device)
        fold_metrics = compute_metrics(all_probs, all_labels_val, threshold=checkpoint['optimal_threshold'])
        all_fold_metrics.append(fold_metrics)

        print(f"\n  第 {fold} 折最佳结果:")
        print(f"    Sn={fold_metrics['sn']:.4f}  Sp={fold_metrics['sp']:.4f}  Acc={fold_metrics['acc']:.4f}  "
              f"MCC={fold_metrics['mcc']:.4f}  Prec={fold_metrics['precision']:.4f}  F1={fold_metrics['f1']:.4f}  "
              f"AUC={fold_metrics['roc_auc']:.4f}  PR-AUC={fold_metrics['pr_auc']:.4f}")
        print(f"    模型已保存: {best_model_path}")

    # 汇总五折结果
    print(f"\n{'='*70}")
    print(f"  五折交叉验证汇总: {scheme_name}")
    print(f"{'='*70}")

    metric_keys = ['sn', 'sp', 'acc', 'mcc', 'precision', 'f1', 'roc_auc', 'pr_auc']
    metric_names = ['灵敏度 (Sn)', '特异性 (Sp)', '准确率 (Acc)', '马修斯系数 (MCC)',
                    '精确率 (Precision)', 'F1 分数 (F1)', 'ROC-AUC', 'PR-AUC (AP)']

    print(f"\n  {'指标':<20} | {'Fold1':>7} | {'Fold2':>7} | {'Fold3':>7} | {'Fold4':>7} | {'Fold5':>7} | {'平均值±标准差':>16}")
    print(f"  {'-'*105}")

    for key, name in zip(metric_keys, metric_names):
        values = [m[key] for m in all_fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        vals_str = ' | '.join([f'{v:7.4f}' for v in values])
        print(f"  {name:<20} | {vals_str} | {mean_val:.4f}±{std_val:.4f}")

    import pandas as pd
    rows = []
    for key, name in zip(metric_keys, metric_names):
        values = [m[key] for m in all_fold_metrics]
        row = {'指标': name}
        for i, v in enumerate(values):
            row[f'Fold{i+1}'] = round(v, 4)
        row['平均值'] = round(np.mean(values), 4)
        row['标准差'] = round(np.std(values), 4)
        row['平均值±标准差'] = f"{np.mean(values):.4f}±{np.std(values):.4f}"
        rows.append(row)

    csv_path = f"5fold_results_{scheme_key}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  结果已保存至: {csv_path}")

    return all_fold_metrics


# ============== 主程序 ==============
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python main_5fold.py <方案名>")
        print("可选方案:")
        for key, val in SCHEMES.items():
            print(f"  {key:30s} -> {val['name']}")
        print("\n示例:")
        print("  python main_5fold.py ban")
        print("  python main_5fold.py cross_attention")
        print("\n运行全部方案:")
        print("  python main_5fold.py all")
        sys.exit(0)

    target = sys.argv[1]

    if target == 'all':
        data = load_all_data()
        all_results = {}
        for scheme_key in SCHEMES:
            results = run_5fold(scheme_key, preloaded_data=data)
            all_results[scheme_key] = results

        metric_keys = ['sn', 'sp', 'acc', 'mcc', 'precision', 'f1', 'roc_auc', 'pr_auc']
        metric_names = ['灵敏度 (Sn)', '特异性 (Sp)', '准确率 (Acc)', '马修斯系数 (MCC)',
                        '精确率 (Precision)', 'F1 分数 (F1)', 'ROC-AUC', 'PR-AUC (AP)']

        def print_summary_table(title, scheme_keys):
            col_w = 18
            print("\n" + "=" * (32 + (col_w + 3) * len(metric_keys)))
            print(f"  {title}")
            print("=" * (32 + (col_w + 3) * len(metric_keys)))
            header = f"  {'模型':<30}"
            for mname in metric_names:
                header += f" | {mname:^{col_w}}"
            print(header)
            print("  " + "-" * (30 + (col_w + 3) * len(metric_keys)))
            for sk in scheme_keys:
                row = f"  {SCHEMES[sk]['name']:<30}"
                for mkey in metric_keys:
                    values = [m[mkey] for m in all_results[sk]]
                    row += f" | {np.mean(values):.4f}±{np.std(values):.4f}"
                print(row)
            print()

        # 表1: 消融实验
        print_summary_table(
            "消融实验 — 五折交叉验证结果 (平均值±标准差)",
            ['without_sequence_encoder', 'without_structure_encoder', 'ban']
        )

        # 表2: 特征融合对比实验
        print_summary_table(
            "特征融合对比实验 — 五折交叉验证结果 (平均值±标准差)",
            ['cross_attention', 'information_bottleneck', 'concatenation', 'ban']
        )

        # 保存汇总CSV
        import pandas as pd
        ablation_keys = ['without_sequence_encoder', 'without_structure_encoder', 'ban']
        fusion_keys = ['cross_attention', 'information_bottleneck', 'concatenation', 'ban']

        for table_name, skeys in [('ablation', ablation_keys), ('fusion_comparison', fusion_keys)]:
            rows = []
            for sk in skeys:
                row = {'模型': SCHEMES[sk]['name']}
                for mkey, mname in zip(metric_keys, metric_names):
                    values = [m[mkey] for m in all_results[sk]]
                    row[mname] = f"{np.mean(values):.4f}±{np.std(values):.4f}"
                rows.append(row)
            csv_path = f"5fold_summary_{table_name}.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  汇总表已保存: {csv_path}")
    else:
        if target not in SCHEMES:
            print(f"未知方案: {target}")
            print(f"可选: {', '.join(SCHEMES.keys())}, all")
            sys.exit(1)
        run_5fold(target)
