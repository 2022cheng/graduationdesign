import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
try:
    # 尝试加载同级目录下的simsun.ttc
    font_path = 'simsun.ttc'
    if os.path.exists(font_path):
        simsun = FontProperties(fname=font_path)
        print(f"成功加载字体文件: {font_path}")
    else:
        # 使用系统自带宋体
        simsun = FontProperties(family='SimSun')
        print("使用系统自带宋体")
except Exception as e:
    print(f"字体设置失败: {e}, 使用默认字体")
    simsun = FontProperties()

# 全局设置字体
plt.rcParams['font.family'] = simsun.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建对比图保存目录
os.makedirs('comparison', exist_ok=True)

# 图像对列表 (01文件夹中的文件名: 02文件夹中的文件名)
image_pairs = {
    'feature_importance.png': 'xgb_feature_importance.png',
    'actual_vs_predicted.png': 'xgb_actual_vs_predicted.png',
    'residual_distribution.png': 'xgb_residual_distribution.png',
    'experience_salary.png': 'xgb_experience_salary.png',
    'partial_dependence.png': 'xgb_partial_dependence.png',
    'education_salary.png': 'xgb_education_salary.png',
    'hyperparameter_heatmap.png': 'xgb_hyperparameter_heatmap.png'
}

# 设置对比图的标题
titles = {
    'feature_importance.png': '特征重要性对比',
    'actual_vs_predicted.png': '实际vs预测薪资对比',
    'residual_distribution.png': '残差分布对比',
    'experience_salary.png': '工作经验与薪资关系对比',
    'partial_dependence.png': '部分依赖图对比',
    'education_salary.png': '教育程度与薪资关系对比',
    'hyperparameter_heatmap.png': '参数优化热力图对比'
}

# 创建并保存对比图
for img1, img2 in image_pairs.items():
    # 读取图像
    img_path1 = os.path.join('01', img1)
    img_path2 = os.path.join('02', img2)

    if not os.path.exists(img_path1):
        print(f"警告: 缺少图像文件 - {img_path1}")
        continue
    if not os.path.exists(img_path2):
        print(f"警告: 缺少图像文件 - {img_path2}")
        continue

    img1_data = mpimg.imread(img_path1)
    img2_data = mpimg.imread(img_path2)

    # 创建对比图
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.suptitle(titles.get(img1, '模型对比'), fontproperties=simsun, fontsize=22, fontweight='bold')

    # 添加左侧图像
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img1_data)
    ax1.set_title('随机森林模型', fontproperties=simsun, fontsize=18, fontweight='bold')
    ax1.axis('off')

    # 添加右侧图像
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(img2_data)
    ax2.set_title('XGBoost模型', fontproperties=simsun, fontsize=18, fontweight='bold')
    ax2.axis('off')

    # 添加整体说明
    plt.figtext(0.5, 0.02,
                "左侧: 随机森林模型结果 | 右侧: XGBoost模型结果",
                ha="center", fontproperties=simsun, fontsize=16,
                bbox={"facecolor": "orange", "alpha": 0.2, "pad": 8})

    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为标题和底部说明留出空间

    # 保存对比图
    comparison_path = os.path.join('comparison', f'comparison_{img1}')
    plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"已保存对比图: {comparison_path}")

print("\n所有对比图已保存到 'comparison' 文件夹")