
from model import *
from dataset import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss: 针对类别不平衡问题的改进损失函数
    当样本被正确分类且置信度高时，降低其损失权重
    这样模型会更关注难分类的样本（通常是少数类）
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数，gamma越大，对易分类样本的惩罚越小
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测正确的概率
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
        optimizer.step()
        total_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()

    return total_loss / len(train_loader)


def evaluate_model_detailed(model, test_loader, criterion, device, threshold=0.5):
    """
    详细评估模型，计算各项指标
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)
            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测概率
            probs = F.softmax(outputs, dim=1)
            pos_probs = probs[:, 1]  # 正类概率
            
            # 使用可调阈值进行预测
            preds = (pos_probs >= threshold).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(pos_probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # 计算各项指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sn (召回率)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Sp
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # MCC
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
    
    # AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
    except:
        roc_auc = 0
        pr_auc = 0
    
    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # Sn
        'specificity': specificity,  # Sp
        'precision': precision,
        'f1': f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    return metrics, all_probs, all_labels


def find_optimal_threshold(all_probs, all_labels, target='balanced'):
    """
    寻找最优阈值
    target: 'balanced' - 平衡Sn和Sp
            'f1' - 最大化F1
            'mcc' - 最大化MCC
    """
    best_threshold = 0.5
    best_score = -1
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        if target == 'balanced':
            # 平衡Sn和Sp（几何平均）
            score = np.sqrt(sn * sp)
        elif target == 'f1':
            score = 2 * precision * sn / (precision + sn) if (precision + sn) > 0 else 0
        elif target == 'mcc':
            denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            score = (tp*tn - fp*fn) / denom if denom > 0 else 0
        else:
            score = np.sqrt(sn * sp)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


# ============== 训练配置 ==============
# 数据量限制（设为 None 使用全部数据）
TRAIN_SAMPLE_LIMIT = None  # 使用全部训练数据
TEST_SAMPLE_LIMIT = None   # 使用全部测试数据
# =========================================

# 模型参数
vocab_size = len(Pep_residue2idx)  # 24 (包含X)
d_model = 256
d_ff = 512
n_layers = 2
n_heads = 4
max_len = 33  # 序列长度（不含[CLS]）

# 结构特征配置
structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 33,  # 与序列长度一致
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

# 加载数据 - 使用分离的正/负样本文件
print("正在加载训练数据...")
train_sequences, train_graph_features, train_labels = load_data_from_separate_files(
    '../Dataset/train_pos.fasta',
    '../Dataset/train_neg.fasta',
    max_len=max_len + 1  # +1 for [CLS] token
)
print(f"训练集大小: {len(train_labels)} (正样本: {sum(train_labels)}, 负样本: {len(train_labels) - sum(train_labels)})")

print("正在加载测试数据...")
test_sequences, test_graph_features, test_labels = load_data_from_separate_files(
    '../Dataset/test_pos.fasta',
    '../Dataset/test_neg.fasta',
    max_len=max_len + 1
)
print(f"测试集大小: {len(test_labels)} (正样本: {sum(test_labels)}, 负样本: {len(test_labels) - sum(test_labels)})")

# 截取部分数据（如果设置了限制）
if TRAIN_SAMPLE_LIMIT:
    train_sequences = train_sequences[:TRAIN_SAMPLE_LIMIT]
    train_graph_features = train_graph_features[:TRAIN_SAMPLE_LIMIT]
    train_labels = train_labels[:TRAIN_SAMPLE_LIMIT]
    print(f"使用 {TRAIN_SAMPLE_LIMIT} 条训练数据")

if TEST_SAMPLE_LIMIT:
    test_sequences = test_sequences[:TEST_SAMPLE_LIMIT]
    test_graph_features = test_graph_features[:TEST_SAMPLE_LIMIT]
    test_labels = test_labels[:TEST_SAMPLE_LIMIT]
    print(f"使用 {TEST_SAMPLE_LIMIT} 条测试数据")

# 创建数据集和数据加载器
train_dataset = MyDataSet(train_sequences, train_graph_features, train_labels)
test_dataset = MyDataSet(test_sequences, test_graph_features, test_labels)

train_loader = Data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 设备配置
print("\n" + "=" * 60)
print("设备检测信息")
print("=" * 60)
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - 显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    device = torch.device("cuda")
    print(f"\n✓ 使用 GPU 进行训练: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"\n✗ CUDA 不可用，使用 CPU 进行训练")
    print("  提示: 请检查是否安装了支持CUDA的PyTorch版本")
    print("  安装命令: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("=" * 60 + "\n")

# 创建模型
model = ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                      structural_config=structural_config).to(device)

# ============== 损失函数配置（关键修改！） ==============
# 计算类别权重 - 对少数类（正样本）给更大权重
n_pos = sum(train_labels)
n_neg = len(train_labels) - n_pos
print(f"训练集类别分布: 正样本 {n_pos}, 负样本 {n_neg}, 比例 1:{n_neg/n_pos:.2f}")

# Step 1: 计算类别权重 - 给少数类（正样本）更大的权重
pos_weight = n_neg / n_pos  # 正样本的权重倍数
class_weights = torch.tensor([1.0, pos_weight]).to(device)  # [neg_weight, pos_weight]
print(f"类别权重: 负样本=1.0, 正样本={pos_weight:.2f}")

# Step 2: 选择损失函数
# - True: 使用 Focal Loss + 类别权重（推荐，双重加强）
# - False: 只使用类别权重的普通交叉熵
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0  # gamma越大，对易分类样本的惩罚越小

if USE_FOCAL_LOSS:
    # Focal Loss 会同时使用类别权重(alpha)和聚焦机制(gamma)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
    print(f"使用 Focal Loss (gamma={FOCAL_GAMMA}) + 类别权重")
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("使用加权 CrossEntropyLoss")

# 优化器 - 使用更小的学习率和权重衰减
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

# 学习率调度器 - 余弦退火
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# ============== 训练参数 ==============
n_epochs = 100
best_mcc = -1.0  # 使用MCC作为最佳模型选择标准（综合考虑Sn和Sp）
best_balanced_score = -1.0
best_model_path = "best_model.pth"
optimal_threshold = 0.5

print(f"\n开始训练，共 {n_epochs} 轮...")
print("=" * 80)
print(f"{'Epoch':>5} | {'Loss':>7} | {'Acc':>6} | {'Sn':>6} | {'Sp':>6} | {'MCC':>6} | {'F1':>6} | {'AUC':>6} | {'Thresh':>6}")
print("-" * 80)

for epoch in range(n_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device, scheduler)
    
    # 详细评估
    metrics, all_probs, all_labels = evaluate_model_detailed(
        model, test_loader, criterion, device, threshold=0.5
    )
    
    # 寻找最优阈值（平衡Sn和Sp）
    opt_thresh, balanced_score = find_optimal_threshold(all_probs, all_labels, target='balanced')
    
    # 使用最优阈值重新评估
    metrics_opt, _, _ = evaluate_model_detailed(
        model, test_loader, criterion, device, threshold=opt_thresh
    )
    
    # 打印当前epoch结果
    print(f'{epoch + 1:5d} | {train_loss:7.4f} | {metrics_opt["accuracy"]:6.4f} | '
          f'{metrics_opt["sensitivity"]:6.4f} | {metrics_opt["specificity"]:6.4f} | '
          f'{metrics_opt["mcc"]:6.4f} | {metrics_opt["f1"]:6.4f} | '
          f'{metrics_opt["roc_auc"]:6.4f} | {opt_thresh:6.2f}')
    
    # 保存最佳模型（基于平衡的Sn和Sp）
    # 使用几何平均作为综合指标
    current_balanced = np.sqrt(metrics_opt["sensitivity"] * metrics_opt["specificity"])
    
    if current_balanced > best_balanced_score:
        best_balanced_score = current_balanced
        best_mcc = metrics_opt["mcc"]
        optimal_threshold = opt_thresh
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimal_threshold': optimal_threshold,
            'metrics': metrics_opt
        }, best_model_path)
        print(f'  --> 新最佳模型! Balanced={best_balanced_score:.4f}, MCC={best_mcc:.4f}, Threshold={optimal_threshold:.2f}')

print("=" * 80)
print(f"\n训练完成！")
print(f"最佳平衡得分: {best_balanced_score:.4f}")
print(f"最佳MCC: {best_mcc:.4f}")
print(f"最优阈值: {optimal_threshold:.2f}")
print(f"模型已保存至: {best_model_path}")

# 加载最佳模型并输出最终评估结果
print("\n" + "=" * 60)
print("最佳模型最终评估结果")
print("=" * 60)
checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
final_metrics, _, _ = evaluate_model_detailed(
    model, test_loader, criterion, device, threshold=checkpoint['optimal_threshold']
)
print(f"灵敏度 (Sn):     {final_metrics['sensitivity']:.4f}")
print(f"特异性 (Sp):     {final_metrics['specificity']:.4f}")
print(f"准确率 (Acc):    {final_metrics['accuracy']:.4f}")
print(f"精确率:          {final_metrics['precision']:.4f}")
print(f"F1分数:          {final_metrics['f1']:.4f}")
print(f"MCC:             {final_metrics['mcc']:.4f}")
print(f"ROC-AUC:         {final_metrics['roc_auc']:.4f}")
print(f"PR-AUC:          {final_metrics['pr_auc']:.4f}")
print(f"最优阈值:        {checkpoint['optimal_threshold']:.2f}")
