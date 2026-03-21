

from model import *
from dataset import *
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
import pandas as pd


def evaluate_model(model, test_loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)

            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / len(test_loader), all_preds, all_labels, all_probs


def calculate_metrics(all_labels, all_preds, all_probs):
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * pre * sn / (pre + sn) if (pre + sn) > 0 else 0
    mcc = matthews_corrcoef(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    return sn, sp, acc, mcc, pre, f1, roc_auc, pr_auc


def save_metrics_to_csv(metrics, filename="metrics.csv"):
    df = pd.DataFrame([metrics])
    df.to_csv(filename, index=False)
    print(f"指标已保存至: {filename}")


# ============== 消融评估: Without Sequence Encoder ==============
print("=" * 60)
print("消融评估: Without Sequence Encoder (仅 Structure Encoder)")
print("=" * 60)

max_len = 33

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

vocab_size = len(Pep_residue2idx)
d_model, d_ff, n_layers, n_heads = 256, 512, 2, 4
structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 33,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

# 加载测试数据
print("正在加载测试数据...")
test_sequences, test_graph_features, test_labels = load_data_from_separate_files(
    '../Dataset/test_pos.fasta',
    '../Dataset/test_neg.fasta',
    max_len=max_len + 1
)
print(f"测试集大小: {len(test_labels)} (正样本: {sum(test_labels)}, 负样本: {len(test_labels) - sum(test_labels)})")

test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载消融模型
model = ToxiPep_WithoutSequenceEncoder(d_model=d_model, structural_config=structural_config).to(device)

checkpoint = torch.load("best_model_without_sequence_encoder.pth", map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
    print(f"已加载模型: best_model_without_sequence_encoder.pth (最优阈值: {optimal_threshold:.2f})")
else:
    model.load_state_dict(checkpoint)
    optimal_threshold = 0.5
    print("已加载模型: best_model_without_sequence_encoder.pth (使用默认阈值: 0.5)")

criterion = nn.CrossEntropyLoss()

# 评估
print(f"\n正在评估模型（阈值: {optimal_threshold:.2f}）...")
_, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion, device, threshold=optimal_threshold)

sn, sp, acc, mcc, pre, f1, roc_auc, pr_auc = calculate_metrics(all_labels, all_preds, all_probs)

print("\n" + "=" * 40)
print("测试集评估结果 (Without Sequence Encoder)")
print("=" * 40)
print(f"  灵敏度 (Sn):        {sn:.4f}")
print(f"  特异性 (Sp):        {sp:.4f}")
print(f"  准确率 (Acc):       {acc:.4f}")
print(f"  马修斯系数 (MCC):   {mcc:.4f}")
print(f"  精确率 (Precision): {pre:.4f}")
print(f"  F1 分数 (F1):       {f1:.4f}")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  PR-AUC (AP):        {pr_auc:.4f}")
print("=" * 40)

metrics = {
    "Sn": sn, "Sp": sp, "Acc": acc, "MCC": mcc,
    "Precision": pre, "F1": f1, "ROC-AUC": roc_auc, "PR-AUC": pr_auc
}
save_metrics_to_csv(metrics, "test_metrics_without_sequence_encoder.csv")
