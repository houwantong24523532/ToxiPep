

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


print("=" * 60)
print("对比评估: Information Bottleneck (DVIB) 特征融合")
print("=" * 60)

max_len = 33
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

vocab_size = len(Pep_residue2idx)
d_model, d_ff, n_layers, n_heads = 256, 512, 2, 4
structural_config = {
    "embedding_dim": 21, "max_seq_len": 33, "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

print("正在加载测试数据...")
test_sequences, test_graph_features, test_labels = load_data_from_separate_files(
    '../Dataset/test_pos.fasta', '../Dataset/test_neg.fasta', max_len=max_len + 1)
print(f"测试集大小: {len(test_labels)}")

test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = ToxiPep_InformationBottleneck(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                                      structural_config=structural_config, k=128).to(device)
checkpoint = torch.load("best_model_information_bottleneck.pth", map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
else:
    model.load_state_dict(checkpoint)
    optimal_threshold = 0.5

criterion = nn.CrossEntropyLoss()
_, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion, device, threshold=optimal_threshold)
sn, sp, acc, mcc, pre, f1, roc_auc, pr_auc = calculate_metrics(all_labels, all_preds, all_probs)

print(f"\n{'='*40}\n测试集评估结果 (Information Bottleneck)\n{'='*40}")
print(f"  Sn: {sn:.4f}  Sp: {sp:.4f}  Acc: {acc:.4f}  MCC: {mcc:.4f}")
print(f"  Precision: {pre:.4f}  F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")

metrics = {"Sn": sn, "Sp": sp, "Acc": acc, "MCC": mcc, "Precision": pre, "F1": f1, "ROC-AUC": roc_auc, "PR-AUC": pr_auc}
save_metrics_to_csv(metrics, "test_metrics_information_bottleneck.csv")
