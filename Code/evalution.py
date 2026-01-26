

from model import *
from dataset import *
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
import pandas as pd


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)

            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]  # 正类概率
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy, all_preds, all_labels, all_probs


def calculate_metrics(all_labels, all_preds, all_probs):
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)  # Sensitivity / Recall
    spe = tn / (tn + fp)  # Specificity
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) > 0 else 0  # F1-Score
    mcc = matthews_corrcoef(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)  # ROC-AUC
    pr_auc = average_precision_score(all_labels, all_probs)  # PR-AUC (Average Precision)
    return acc, sen, spe, pre, f1, mcc, roc_auc, pr_auc


def save_metrics_to_csv(metrics, filename="metrics.csv"):
    df = pd.DataFrame([metrics])
    df.to_csv(filename, index=False)
    print(f"指标已保存至: {filename}")


# ============== 配置参数 ==============
max_len = 33  # 序列长度（不含[CLS]）

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型参数
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

# 加载模型
model = ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                      structural_config=structural_config).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
print("已加载模型: best_model.pth")

criterion = nn.CrossEntropyLoss()

# 评估模型
print("\n正在评估模型...")
_, _, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion, device)

# 计算指标
acc, sen, spe, pre, f1, mcc, roc_auc, pr_auc = calculate_metrics(all_labels, all_preds, all_probs)

# 打印结果
print("\n" + "=" * 60)
print("测试集评估结果")
print("=" * 60)
print(f"  准确率 (Acc):       {acc:.4f}")
print(f"  灵敏度 (Sn):        {sen:.4f}")
print(f"  特异性 (Sp):        {spe:.4f}")
print(f"  精确率 (Precision): {pre:.4f}")
print(f"  F1分数 (F1):        {f1:.4f}")
print(f"  马修斯系数 (MCC):   {mcc:.4f}")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  PR-AUC (AP):        {pr_auc:.4f}")
print("=" * 60)

# 保存指标
metrics = {
    "Acc": acc, 
    "Sn": sen, 
    "Sp": spe, 
    "Precision": pre,
    "F1": f1,
    "MCC": mcc, 
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc
}
save_metrics_to_csv(metrics, "test_metrics.csv")
