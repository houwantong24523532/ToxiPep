# merge_figures.py - 把已生成的图片拼接
# DeepKsite 项目

import os
from PIL import Image


def merge_images(image_paths, cols, save_path):
    imgs = [Image.open(p) for p in image_paths]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols

    merged = Image.new('RGB', (w * cols, h * rows), 'white')
    for i, img in enumerate(imgs):
        r, c = i // cols, i % cols
        img = img.resize((w, h))
        merged.paste(img, (c * w, r * h))

    merged.save(save_path, dpi=(300, 300))
    print(f"  已保存: {save_path}")


base_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(base_dir, '..', 'figures')
scatter_dir = os.path.join(figures_dir, 'scatter')
bar_dir = os.path.join(figures_dir, 'bar')

# 测试集散点图 6合1
test_scatter = [os.path.join(scatter_dir, f'scatter_test_{n}.png')
                for n in ['AAC', 'PAAC', 'CKSAAGP', 'DPC', 'PHYC', 'DeepKsite']]
merge_images(test_scatter, 3, os.path.join(figures_dir, 'scatter_combined_test.png'))

# 训练集散点图 6合1
train_scatter = [os.path.join(scatter_dir, f'scatter_train_{n}.png')
                 for n in ['AAC', 'PAAC', 'CKSAAGP', 'DPC', 'PHYC', 'DeepKsite']]
merge_images(train_scatter, 3, os.path.join(figures_dir, 'scatter_combined_train.png'))

# 柱状图 6合1
bar_files = [os.path.join(bar_dir, f'bar_{n}.png')
             for n in ['Logistic_Regression', 'KNN', 'Random_Forest', 'SVM', 'Naive_Bayes', 'Gradient_Boosting']]
merge_images(bar_files, 3, os.path.join(figures_dir, 'bar_combined.png'))

print("\n  全部拼接完成！")
