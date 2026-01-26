

from model import *
from dataset import *

def train_model(model, train_loader, criterion, optimizer, device):
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

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, graph_features, labels = batch
            input_ids, graph_features, labels = input_ids.to(device), graph_features.to(device), labels.to(device)
            outputs = model(input_ids, graph_features, device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy


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

# 损失函数和优化器
# 使用加权损失函数来处理类别不平衡问题
n_pos = sum(train_labels)
n_neg = len(train_labels) - n_pos
weight = torch.tensor([n_pos / len(train_labels), n_neg / len(train_labels)]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = optim.AdamW(model.parameters(), lr=0.0005)

# 训练参数
n_epochs = 100
best_accuracy = 0.0
best_model_path = "best_model.pth"

print(f"\n开始训练，共 {n_epochs} 轮...")
print("=" * 60)

for epoch in range(n_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

    print(f'Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}')

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'  --> 新最佳模型保存! 准确度: {best_accuracy:.4f}')

print("=" * 60)
print(f"训练完成！最佳准确度: {best_accuracy:.4f}")
print(f"模型已保存至: {best_model_path}")
