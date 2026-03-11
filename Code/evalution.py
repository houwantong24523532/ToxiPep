

from model import *
from dataset import *
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
import pandas as pd


def evaluate_model(model, test_loader, criterion, device, threshold=0.5):
    """
    评估模型，支持自定义阈值
    """
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
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) > 0 else 0
    mcc = matthews_corrcoef(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    return mcc, f1, roc_auc, pr_auc


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

# 加载checkpoint（新格式包含 model_state_dict, optimal_threshold, metrics）
checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)

# 兼容新旧两种保存格式
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    # 新格式：包含额外信息
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
    print(f"已加载模型: best_model.pth (最优阈值: {optimal_threshold:.2f})")
else:
    # 旧格式：直接是 state_dict
    model.load_state_dict(checkpoint)
    optimal_threshold = 0.5
    print("已加载模型: best_model.pth (使用默认阈值: 0.5)")

criterion = nn.CrossEntropyLoss()

# 评估模型（使用最优阈值）
print(f"\n正在评估模型（阈值: {optimal_threshold:.2f}）...")
_, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion, device, threshold=optimal_threshold)

# 计算指标
mcc, f1, roc_auc, pr_auc = calculate_metrics(all_labels, all_preds, all_probs)

# 打印结果
print("\n" + "=" * 40)
print("测试集评估结果")
print("=" * 40)
print(f"  马修斯系数 (MCC):   {mcc:.4f}")
print(f"  F1分数 (F1):        {f1:.4f}")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  PR-AUC (AP):        {pr_auc:.4f}")
print("=" * 40)

# 保存指标
metrics = {
    "MCC": mcc,
    "F1": f1,
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc
}
save_metrics_to_csv(metrics, "test_metrics.csv")
