import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, matthews_corrcoef, roc_auc_score
from model import ToxiPep_Model
from dataset import convert_to_graph_channel


# 残基到索引的映射 (包含X未知氨基酸)
Pep_residue2idx = {
    '[PAD]': 0, '[CLS]': 1, '[SEP]': 2,
    'A': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7,
    'G': 8, 'H': 9, 'I': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22,
    'X': 23  # 未知氨基酸
}


def read_fasta(file_path):
    """读取FASTA格式文件"""
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq.upper())
                    seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq.upper())
    return sequences


def transform_Pep_to_index(sequences, residue2idx):
    """将氨基酸序列转换为索引"""
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, residue2idx['X']) for residue in seq]
        token_index.append(seq_id)
    return token_index


def pad_sequence(token_list, max_len=34):
    """填充序列到固定长度"""
    data = []
    for seq in token_list:
        seq = [Pep_residue2idx['[CLS]']] + seq
        n_pad = max_len - len(seq)
        seq.extend([Pep_residue2idx['[PAD]']] * max(n_pad, 0))
        data.append(seq[:max_len])
    return data


class PeptideDataset(Dataset):
    def __init__(self, sequences, graph_features):
        self.sequences = sequences
        self.graph_features = graph_features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]), torch.tensor(self.graph_features[idx]))


def evaluate_model(model, data_loader, device):
    """模型预测"""
    model.eval()
    predictions, probabilities = [], []
    with torch.no_grad():
        for input_ids, graph_features in data_loader:
            input_ids, graph_features = input_ids.to(device), graph_features.to(device)
            outputs = model(input_ids, graph_features, device)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类概率
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    return predictions, probabilities


# 命令行参数
parser = argparse.ArgumentParser(description='蛋白质修饰预测')
parser.add_argument('-i', type=str, required=True, help='输入FASTA文件')
parser.add_argument('-o', type=str, required=True, help='输出预测结果文件')
parser.add_argument('-m', type=str, default='best_model.pth', help='模型文件路径 (默认: best_model.pth)')
args = parser.parse_args()

# 读取输入序列
sequences = read_fasta(args.i)
print(f"读取到 {len(sequences)} 条序列")

# 模型参数配置
max_len = 33  # 序列长度（不含[CLS]）
vocab_size = len(Pep_residue2idx)
d_model, d_ff, n_layers, n_heads = 256, 512, 2, 4

structural_config = {
    "embedding_dim": 21,
    "max_seq_len": 33,
    "filter_num": 64,
    "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)]
}

# 处理序列
indexed_sequences = transform_Pep_to_index(sequences, Pep_residue2idx)
padded_sequences = pad_sequence(indexed_sequences, max_len=max_len + 1)
graph_features = [convert_to_graph_channel(seq, max_seq_len=max_len) for seq in sequences]

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, max_len, structural_config=structural_config).to(device)
model.load_state_dict(torch.load(args.m, map_location=device))
print(f"已加载模型: {args.m}")

# 创建数据集和数据加载器
test_dataset = PeptideDataset(padded_sequences, graph_features)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 进行预测
predictions, probabilities = evaluate_model(model, test_loader, device)

# 保存结果
idx_to_residue = {v: k for k, v in Pep_residue2idx.items()}

with open(args.o, 'w') as f:
    f.write("序列\t预测结果\t置信度\n")
    for seq, pred, prob in zip(sequences, predictions, probabilities):
        result = "已修饰" if pred == 1 else "未修饰"
        confidence = prob if pred == 1 else 1 - prob
        f.write(f"{seq}\t{result}\t{confidence:.6f}\n")

print(f"预测结果已保存至: {args.o}")
print(f"已修饰: {sum(predictions)} 条")
print(f"未修饰: {len(predictions) - sum(predictions)} 条")
