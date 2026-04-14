# features.py - 特征提取函数
# DeepKsite 项目

import numpy as np
from tqdm import tqdm

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


def read_fasta(pos_file, neg_file):
    """从正负样本fasta文件读取序列和标签"""
    sequences, labels = [], []
    for filepath, label in [(pos_file, 1), (neg_file, 0)]:
        with open(filepath, 'r') as f:
            current_seq = ''
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        labels.append(label)
                    current_seq = ''
                else:
                    current_seq += line.upper()
            if current_seq:
                sequences.append(current_seq)
                labels.append(label)
    return sequences, np.array(labels)


def extract_aac(sequences):
    """AAC: 氨基酸组成，20维"""
    features = []
    for seq in tqdm(sequences, desc="  AAC"):
        length = max(len(seq), 1)
        feat = [seq.count(aa) / length for aa in AMINO_ACIDS]
        features.append(feat)
    return np.array(features)


def extract_dpc(sequences):
    """DPC: 二肽组成，400维"""
    dipeptides = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
    features = []
    for seq in tqdm(sequences, desc="  DPC"):
        length = max(len(seq) - 1, 1)
        feat = []
        for dp in dipeptides:
            count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == dp)
            feat.append(count / length)
        features.append(feat)
    return np.array(features)


def extract_paac(sequences, lambda_val=5, w=0.05):
    """PAAC: 伪氨基酸组成，(20+lambda)维"""
    hydrophobicity = {
        'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
        'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
        'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
        'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
    }
    hydrophilicity = {
        'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5,
        'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8,
        'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0,
        'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3
    }
    mass = {
        'A': 15, 'C': 47, 'D': 59, 'E': 73, 'F': 91, 'G': 1,
        'H': 82, 'I': 57, 'K': 73, 'L': 57, 'M': 75, 'N': 58,
        'P': 42, 'Q': 72, 'R': 101, 'S': 31, 'T': 45, 'V': 43,
        'W': 130, 'Y': 107
    }

    def theta(seq, prop, k):
        vals = [prop.get(aa, 0) for aa in seq]
        if len(vals) <= k:
            return 0
        return sum((vals[i] - vals[i+k])**2 for i in range(len(vals)-k)) / (len(vals)-k)

    features = []
    for seq in tqdm(sequences, desc="  PAAC"):
        aac = [seq.count(aa) / max(len(seq), 1) for aa in AMINO_ACIDS]
        thetas = []
        for lam in range(1, lambda_val + 1):
            t1 = theta(seq, hydrophobicity, lam)
            t2 = theta(seq, hydrophilicity, lam)
            t3 = theta(seq, mass, lam)
            thetas.append((t1 + t2 + t3) / 3)
        denom = 1 + w * sum(thetas)
        paac_aac = [v / denom for v in aac]
        paac_theta = [w * t / denom for t in thetas]
        features.append(paac_aac + paac_theta)
    return np.array(features)


def extract_cksaagp(sequences, max_k=2):
    """CKSAAGP: k间隔氨基酸组对频率"""
    groups = {
        'G1': set('AGV'),
        'G2': set('ILFP'),
        'G3': set('YMTS'),
        'G4': set('HNQW'),
        'G5': set('RK'),
        'G6': set('DE'),
        'G7': set('C'),
    }
    group_names = sorted(groups.keys())

    def get_group(aa):
        for gname, gset in groups.items():
            if aa in gset:
                return gname
        return None

    features = []
    for seq in tqdm(sequences, desc="  CKSAAGP"):
        feat = []
        for k in range(max_k + 1):
            pairs = {}
            for g1 in group_names:
                for g2 in group_names:
                    pairs[(g1, g2)] = 0
            total = 0
            for i in range(len(seq) - k - 1):
                g1 = get_group(seq[i])
                g2 = get_group(seq[i + k + 1])
                if g1 and g2:
                    pairs[(g1, g2)] += 1
                    total += 1
            for g1 in group_names:
                for g2 in group_names:
                    feat.append(pairs[(g1, g2)] / max(total, 1))
        features.append(feat)
    return np.array(features)


def extract_phyc(sequences):
    """PHYC: 理化性质特征，10维"""
    hydro = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    charge = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
    mw = {
        'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165, 'G': 75,
        'H': 155, 'I': 131, 'K': 146, 'L': 131, 'M': 149, 'N': 132,
        'P': 115, 'Q': 146, 'R': 174, 'S': 105, 'T': 119, 'V': 117,
        'W': 204, 'Y': 181
    }
    aromatic = set('FYW')
    charged_aa = set('DEKRH')

    features = []
    for seq in tqdm(sequences, desc="  PHYC"):
        n = max(len(seq), 1)
        length = len(seq) / 50.0
        avg_hydro = np.mean([hydro.get(aa, 0) for aa in seq])
        net_charge = sum(charge.get(aa, 0) for aa in seq)
        total_mw = sum(mw.get(aa, 111) for aa in seq) / (n * 150.0)
        arom_ratio = sum(1 for aa in seq if aa in aromatic) / n
        charge_ratio = sum(1 for aa in seq if aa in charged_aa) / n
        hydro_ratio = sum(1 for aa in seq if hydro.get(aa, 0) > 0) / n
        cys_ratio = seq.count('C') / n
        kr_ratio = (seq.count('K') + seq.count('R')) / n
        pro_ratio = seq.count('P') / n
        features.append([length, avg_hydro, net_charge, total_mw,
                         arom_ratio, charge_ratio, hydro_ratio,
                         cys_ratio, kr_ratio, pro_ratio])
    return np.array(features)


def extract_ban_features(pos_file, neg_file, model_path, device):
    """加载DeepKsite模型，提取BAN层输出特征（只读取，不修改模型）"""
    import sys
    import os
    import torch
    import torch.utils.data as Data
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import load_data_from_separate_files, MyDataSet, Pep_residue2idx
    from model import ToxiPep_Model

    max_len = 34
    vocab_size = len(Pep_residue2idx)
    d_model, d_ff, n_layers, n_heads = 256, 512, 2, 4
    structural_config = {
        "embedding_dim": 21, "max_seq_len": 33,
        "filter_num": 64, "filter_sizes": [(3,3),(5,5),(7,7),(9,9)]
    }

    model = ToxiPep_Model(vocab_size, d_model, d_ff, n_layers, n_heads, 33,
                          structural_config=structural_config).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("  DeepKsite模型已加载（只读模式）")

    seqs, graphs, labels = load_data_from_separate_files(pos_file, neg_file, max_len=max_len)
    dataset = MyDataSet(seqs, graphs, labels)
    loader = Data.DataLoader(dataset, batch_size=8, shuffle=False)

    ban_outputs = []
    def hook_fn(module, input, output):
        ban_outputs.append(output.detach().cpu().numpy())
    hook = model.ban.register_forward_hook(hook_fn)

    print("  提取BAN特征中...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="  BAN"):
            input_ids, graph_features, lbls = batch
            input_ids = input_ids.to(device)
            graph_features = graph_features.to(device)
            _ = model(input_ids, graph_features, device)
            torch.cuda.empty_cache()

    hook.remove()
    ban_features = np.concatenate(ban_outputs, axis=0)
    all_labels = np.array(labels)
    print(f"  BAN特征提取完成: {ban_features.shape}")
    return ban_features, all_labels
