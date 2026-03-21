
from model import *
from dataset import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
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


# ============== 对比实验: Cross Attention ==============
print("=" * 60)
print("对比实验: Cross Attention 特征融合")
print("=" * 60)

TRAIN_SAMPLE_LIMIT = None
TEST_SAMPLE_LIMIT = None

vocab_size = len(Pep_residue2idx)
d_model = 256
d_ff = 512
n_layers = 2
n_heads = 4
max_len = 33

structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 33,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

print("正在加载训练数据...")
train_sequences, train_graph_features, train_labels = load_data_from_separate_files(
    '../Dataset/train_pos.fasta', '../Dataset/train_neg.fasta', max_len=max_len + 1)
print(f"训练集大小: {len(train_labels)} (正样本: {sum(train_labels)}, 负样本: {len(train_labels) - sum(train_labels)})")

print("正在加载测试数据...")
test_sequences, test_graph_features, test_labels = load_data_from_separate_files(
    '../Dataset/test_pos.fasta', '../Dataset/test_neg.fasta', max_len=max_len + 1)
print(f"测试集大小: {len(test_labels)} (正样本: {sum(test_labels)}, 负样本: {len(test_labels) - sum(test_labels)})")

if TRAIN_SAMPLE_LIMIT:
    train_sequences = train_sequences[:TRAIN_SAMPLE_LIMIT]
    train_graph_features = train_graph_features[:TRAIN_SAMPLE_LIMIT]
    train_labels = train_labels[:TRAIN_SAMPLE_LIMIT]
if TEST_SAMPLE_LIMIT:
    test_sequences = test_sequences[:TEST_SAMPLE_LIMIT]
    test_graph_features = test_graph_features[:TEST_SAMPLE_LIMIT]
    test_labels = test_labels[:TEST_SAMPLE_LIMIT]

train_dataset = MyDataSet(train_sequences, train_graph_features, train_labels)
test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)
train_loader = Data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print("\n" + "=" * 60)
print("设备检测信息")
print("=" * 60)
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda")
    print(f"\n使用 GPU 进行训练: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"\nCUDA 不可用，使用 CPU 进行训练")
print("=" * 60 + "\n")

model = ToxiPep_CrossAttention(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                               structural_config=structural_config).to(device)

n_pos = sum(train_labels)
n_neg = len(train_labels) - n_pos
print(f"训练集类别分布: 正样本 {n_pos}, 负样本 {n_neg}, 比例 1:{n_neg/n_pos:.2f}")
pos_weight = np.sqrt(n_neg / n_pos)
class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
print(f"类别权重: 负样本=1.0, 正样本={pos_weight:.2f}")

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
if USE_FOCAL_LOSS:
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
    print(f"使用 Focal Loss (gamma={FOCAL_GAMMA}) + 类别权重")
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

n_epochs = 100
best_score = -1.0
best_model_path = "best_model_cross_attention.pth"
optimal_threshold = 0.5

print(f"\n开始训练，共 {n_epochs} 轮...")
print("=" * 70)
print(f"{'Epoch':>5} | {'Loss':>7} | {'Sn':>6} | {'Sp':>6} | {'Acc':>6} | {'MCC':>6} | {'Prec':>6} | {'F1':>6} | {'AUC':>6} | {'PR-AUC':>6} | {'Thresh':>6}")
print("-" * 110)

for epoch in range(n_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device, scheduler)
    all_probs, all_labels, _ = model_inference(model, test_loader, criterion, device)
    opt_thresh, _ = find_optimal_threshold(all_probs, all_labels)
    metrics_opt = compute_metrics(all_probs, all_labels, threshold=opt_thresh)
    print(f'{epoch + 1:5d} | {train_loss:7.4f} | '
          f'{metrics_opt["sn"]:6.4f} | {metrics_opt["sp"]:6.4f} | {metrics_opt["acc"]:6.4f} | '
          f'{metrics_opt["mcc"]:6.4f} | {metrics_opt["precision"]:6.4f} | {metrics_opt["f1"]:6.4f} | '
          f'{metrics_opt["roc_auc"]:6.4f} | {metrics_opt["pr_auc"]:6.4f} | {opt_thresh:6.2f}')
    composite_score = 0.5 * metrics_opt["mcc"] + 0.5 * metrics_opt["pr_auc"]
    if composite_score > best_score:
        best_score = composite_score
        optimal_threshold = opt_thresh
        torch.save({'model_state_dict': model.state_dict(), 'optimal_threshold': optimal_threshold, 'metrics': metrics_opt}, best_model_path)
        print(f'  --> 新最佳模型! Score={best_score:.4f} (MCC={metrics_opt["mcc"]:.4f}, PR-AUC={metrics_opt["pr_auc"]:.4f}), Threshold={optimal_threshold:.2f}')

print("=" * 110)
print(f"\n训练完成！最佳综合评分: {best_score:.4f}, 最优阈值: {optimal_threshold:.2f}")
print(f"模型已保存至: {best_model_path}")

print("\n" + "=" * 60)
print("最佳模型最终评估结果 (Cross Attention)")
print("=" * 60)
checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
all_probs, all_labels, _ = model_inference(model, test_loader, criterion, device)
final_metrics = compute_metrics(all_probs, all_labels, threshold=checkpoint['optimal_threshold'])
print(f"灵敏度 (Sn):     {final_metrics['sn']:.4f}")
print(f"特异性 (Sp):     {final_metrics['sp']:.4f}")
print(f"准确率 (Acc):    {final_metrics['acc']:.4f}")
print(f"马修斯系数 (MCC): {final_metrics['mcc']:.4f}")
print(f"精确率 (Prec):   {final_metrics['precision']:.4f}")
print(f"F1 分数 (F1):    {final_metrics['f1']:.4f}")
print(f"ROC-AUC:         {final_metrics['roc_auc']:.4f}")
print(f"PR-AUC (AP):     {final_metrics['pr_auc']:.4f}")
print(f"最优阈值:        {checkpoint['optimal_threshold']:.2f}")
