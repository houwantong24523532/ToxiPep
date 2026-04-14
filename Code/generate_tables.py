# generate_tables.py - 生成5张表格图片
# DeepKsite 项目

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def find_best_in_columns(data, metrics_start_col=1, higher_is_better=True):
    """找出每列最大值的行索引"""
    best_rows = {}
    for col in range(metrics_start_col, len(data[0])):
        best_val = None
        best_row = None
        for row in range(len(data)):
            val_str = str(data[row][col]).replace('*', '').strip()
            # 处理 ± 格式
            if '±' in val_str:
                val = float(val_str.split('±')[0])
            else:
                try:
                    val = float(val_str)
                except:
                    continue
            if best_val is None or val > best_val:
                best_val = val
                best_row = row
        best_rows[col] = best_row
    return best_rows


def add_stars(data, metrics_start_col=1):
    """给每列最优值加星号"""
    best_rows = find_best_in_columns(data, metrics_start_col)
    new_data = []
    for r, row in enumerate(data):
        new_row = list(row)
        for col, best_row in best_rows.items():
            if r == best_row:
                val = str(new_row[col]).replace('*', '')
                new_row[col] = val + '*'
        new_data.append(new_row)
    return new_data


def render_table(headers, data, title, save_path, col_widths=None):
    """用matplotlib渲染表格并保存为PNG"""
    n_cols = len(headers)
    n_rows = len(data)

    if col_widths is None:
        col_widths = [0.18] + [0.1] * (n_cols - 1)

    fig_width = sum(col_widths) * 10 + 1
    fig_height = (n_rows + 2) * 0.45 + 0.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15, loc='left')

    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # 样式
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold', fontsize=9)
            cell.set_height(0.08)
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            else:
                cell.set_facecolor('white')
            cell.set_height(0.07)

            # 星号值加粗
            text = cell.get_text().get_text()
            if '*' in text:
                cell.set_text_props(fontweight='bold')

        # 第一列左对齐
        if col == 0:
            cell.set_text_props(ha='left')
            cell._loc = 'left'

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, '..', 'figures', 'tables')
    os.makedirs(output_dir, exist_ok=True)

    # ============ 表格1: 与SOTA方法对比 ============
    headers1 = ['Model', 'ACC', 'MCC', 'Sn', 'Sp', 'AUC', 'F1-Score', 'AP']
    data1 = [
        ['DeepSuccinylSite', '77.39', '18.89', '18.47', '94.25', '73.36', '26.67', '40.61'],
        ['LSTMCNNsucc',      '80.51', '32.30', '28.77', '94.94', '82.23', '39.16', '54.31'],
        ['Deep_Ksucc',       '80.57', '33.52', '31.54', '94.24', '81.45', '41.44', '52.66'],
        ['Deep_KsuccSite',   '75.41', '30.62', '48.94', '82.79', '76.67', '46.46', '45.18'],
        ['LMSuccSite',       '81.30', '36.62', '34.19', '94.43', '82.54', '44.36', '56.38'],
        ['pSuc-EDBAM',       '79.88', '33.84', '36.92', '91.86', '81.47', '44.45', '51.79'],
        ['BioSeq_Ksite',     '77.15', '44.30', '71.34', '78.77', '82.64', '57.65', '54.39'],
        ['DeepKsite',        '79.14', '47.13', '70.81', '81.46', '83.07', '59.68', '57.31'],
    ]
    data1 = add_stars(data1)
    render_table(headers1, data1, 'Table 1. Performance Comparison on Test Dataset',
                 os.path.join(output_dir, 'table1_sota_comparison.png'))

    # ============ 表格2: 测试集消融实验 ============
    headers2 = ['Model', 'Sn', 'Sp', 'Acc', 'MCC', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC']
    data2 = [
        ['w/o Sequence Encoder',  '0.8779', '0.3655', '0.4772', '0.2169', '0.2784', '0.4227', '0.6603', '0.3050'],
        ['w/o Structure Encoder', '0.8457', '0.5534', '0.6172', '0.3304', '0.3456', '0.4907', '0.7572', '0.4160'],
        ['DeepKsite (BAN)',       '0.7081', '0.8146', '0.7914', '0.4713', '0.5157', '0.5968', '0.8307', '0.5731'],
    ]
    data2 = add_stars(data2)
    render_table(headers2, data2, 'Table 2. Ablation Study on Test Dataset',
                 os.path.join(output_dir, 'table2_ablation_test.png'))

    # ============ 表格3: 测试集融合对比 ============
    headers3 = ['Model', 'Sn', 'Sp', 'Acc', 'MCC', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC']
    data3 = [
        ['Cross Attention',       '0.7142', '0.7256', '0.7231', '0.3761', '0.4205', '0.5294', '0.7894', '0.4758'],
        ['Information Bottleneck','0.7654', '0.6819', '0.7001', '0.3747', '0.4015', '0.5267', '0.7939', '0.4807'],
        ['Concatenation',         '0.8116', '0.3835', '0.4768', '0.1699', '0.2685', '0.4035', '0.6191', '0.2710'],
        ['DeepKsite (BAN)',       '0.7081', '0.8146', '0.7914', '0.4713', '0.5157', '0.5968', '0.8307', '0.5731'],
    ]
    data3 = add_stars(data3)
    render_table(headers3, data3, 'Table 3. Fusion Method Comparison on Test Dataset',
                 os.path.join(output_dir, 'table3_fusion_test.png'))

    # ============ 表格4: 5折消融实验 ============
    headers4 = ['Model', 'Sn', 'Sp', 'Acc', 'MCC', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC']
    data4 = [
        ['w/o Sequence Encoder',  '0.8185±0.0972', '0.4384±0.2226', '0.5212±0.1534', '0.2131±0.1070', '0.3025±0.0435', '0.4361±0.0396', '0.6940±0.0468', '0.3564±0.0476'],
        ['w/o Structure Encoder', '0.7506±0.0299', '0.7643±0.0214', '0.7613±0.0107', '0.4469±0.0070', '0.4711±0.0132', '0.5783±0.0051', '0.8344±0.0039', '0.5578±0.0101'],
        ['DeepKsite (BAN)',       '0.7513±0.0322', '0.7727±0.0222', '0.7680±0.0107', '0.4569±0.0064', '0.4805±0.0142', '0.5854±0.0044', '0.8389±0.0048', '0.5702±0.0145'],
    ]
    data4 = add_stars(data4)
    render_table(headers4, data4, 'Table 4. Ablation Study (5-Fold Cross Validation)',
                 os.path.join(output_dir, 'table4_ablation_5fold.png'),
                 col_widths=[0.20] + [0.13] * 8)

    # ============ 表格5: 5折融合对比 ============
    headers5 = ['Model', 'Sn', 'Sp', 'Acc', 'MCC', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC']
    data5 = [
        ['Cross Attention',       '0.7701±0.0118', '0.7560±0.0143', '0.7591±0.0090', '0.4532±0.0080', '0.4685±0.0110', '0.5824±0.0062', '0.8383±0.0044', '0.5682±0.0104'],
        ['Information Bottleneck','0.7369±0.0431', '0.7795±0.0346', '0.7702±0.0177', '0.4540±0.0049', '0.4848±0.0230', '0.5833±0.0048', '0.8384±0.0040', '0.5680±0.0096'],
        ['Concatenation',         '0.7187±0.0193', '0.7909±0.0141', '0.7751±0.0073', '0.4513±0.0065', '0.4898±0.0112', '0.5823±0.0045', '0.8371±0.0050', '0.5676±0.0134'],
        ['DeepKsite (BAN)',       '0.7513±0.0322', '0.7727±0.0222', '0.7680±0.0107', '0.4569±0.0064', '0.4805±0.0142', '0.5854±0.0044', '0.8389±0.0048', '0.5702±0.0145'],
    ]
    data5 = add_stars(data5)
    render_table(headers5, data5, 'Table 5. Fusion Comparison (5-Fold Cross Validation)',
                 os.path.join(output_dir, 'table5_fusion_5fold.png'),
                 col_widths=[0.20] + [0.13] * 8)

    print("\n  全部表格生成完成！")


if __name__ == '__main__':
    main()
