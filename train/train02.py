import pandas as pd
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
import joblib
import xgboost as xgb
import matplotlib as mpl
import os

# 设置中文字体支持
try:
    # 尝试加载同级目录下的simsun.ttc
    font_path = 'simsun.ttc'
    if os.path.exists(font_path):
        font_prop = mpl.font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"成功加载字体文件: {font_path}")
    else:
        # 使用系统自带宋体
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        print("使用系统自带宋体")
except Exception as e:
    print(f"字体设置失败: {e}, 使用默认字体")

# 1. 数据加载与预处理
def load_data(file_path):
    df = pd.read_csv(file_path)

    # 薪资处理：提取薪资中位数
    def parse_salary(s):
        try:
            nums = re.findall(r'\d+', s)
            nums = [int(n) for n in nums]
            return np.median(nums) if nums else np.nan
        except:
            return np.nan

    df['salary'] = df['salary'].apply(parse_salary)

    # 工作经验转换
    exp_map = {
        '经验不限': 0,
        '在校/应届': 0.5,
        '1年以内': 1,
        '1-3年': 2,
        '3-5年': 4,
        '5-10年': 7.5,
        '10年以上': 10
    }
    df['workExperience'] = df['workExperience'].map(exp_map).fillna(0)

    # 公司规模转换
    def parse_company_size(s):
        try:
            nums = re.findall(r'\d+', s)
            if len(nums) == 2:
                return (int(nums[0]) + int(nums[1])) / 2
            elif nums:
                return int(nums[0])
            return 0
        except:
            return 0

    df['companyPeople'] = df['companyPeople'].apply(parse_company_size)

    # 薪资月份处理
    df['salaryMonth'] = df['salaryMonth'].replace('0薪', 0)
    df['salaryMonth'] = pd.to_numeric(df['salaryMonth'], errors='coerce').fillna(12)

    # 标签计数特征
    def count_tags(s):
        try:
            tags = ast.literal_eval(s)
            return len(tags) if isinstance(tags, list) else 0
        except:
            return 0

    df['workTag_count'] = df['workTag'].apply(count_tags)
    df['companyTags_count'] = df['companyTags'].apply(count_tags)

    # 教育程度编码
    edu_map = {'博士': 5, '硕士': 4, '本科': 3, '大专': 2, '学历不限': 1}
    df['educational'] = df['educational'].map(edu_map).fillna(1)

    return df

# 2. 特征工程
def feature_engineering(df):
    # 选择关键特征
    features = df[[
        'educational', 'workExperience', 'salaryMonth',
        'companyPeople', 'workTag_count', 'companyTags_count',
        'address', 'type', 'companyNature'
    ]]

    # 目标变量
    target = df['salary'].dropna()
    features = features.loc[target.index]

    return features, target

# 3. 加载数据
df = load_data('temp3.csv')

# 4. 特征工程
X, y = feature_engineering(df)

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. 预处理管道
categorical_features = ['address', 'type', 'companyNature']
numerical_features = [
    'educational', 'workExperience', 'salaryMonth',
    'companyPeople', 'workTag_count', 'companyTags_count'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 7. 使用XGBoost模型构建与调参
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# 8. 最佳模型评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"最佳参数: {grid_search.best_params_}")
print(f"测试集RMSE: {rmse:.2f}")
print(f"测试集R²: {r2:.4f}")

# 9. 特征重要性分析
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_features = numerical_features + list(cat_features)

# 获取XGBoost特征重要性
xgb_model = best_model.named_steps['regressor']
importances = xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n特征重要性TOP 10:")
print(feature_importance.head(10))

# 10. 保存模型
joblib.dump(best_model, '02/salary_predictor_xgb.pkl')

# ======================================================
# 绘图代码
# ======================================================

# 设置论文级别的绘图风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.bbox': 'tight'
})

# 1. 特征重要性分布图 (水平条形图)
plt.figure()
top_features = feature_importance.head(15).sort_values('Importance', ascending=True)
palette = sns.color_palette("viridis", len(top_features))
plt.barh(top_features['Feature'], top_features['Importance'], color=palette)
plt.xlabel('特征重要性', fontweight='bold')
plt.ylabel('特征名称', fontweight='bold')
plt.title('TOP 15 特征重要性分布 (XGBoost)', fontsize=16, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('xgb_feature_importance.png')
plt.close()

# 2. 实际薪资 vs 预测薪资散点图
plt.figure()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None, color='royalblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('实际薪资 (元)', fontweight='bold')
plt.ylabel('预测薪资 (元)', fontweight='bold')
plt.title(f'实际薪资 vs 预测薪资 (XGBoost, R² = {r2:.4f})', fontsize=14, fontweight='bold')
plt.grid(linestyle='--', alpha=0.5)
plt.text(0.05, 0.95, f'RMSE = {rmse:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
plt.tight_layout()
plt.savefig('xgb_actual_vs_predicted.png')
plt.close()

# 3. 残差分布图
residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True, color='teal', bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('残差 (实际薪资 - 预测薪资)', fontweight='bold')
plt.ylabel('频数', fontweight='bold')
plt.title(f'预测残差分布 (均值 = {residuals.mean():.2f}, 标准差 = {residuals.std():.2f})',
          fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('xgb_residual_distribution.png')
plt.close()

# 4. 工作经验与薪资关系图
plt.figure()
# 247行：
sns.boxplot(x=df['workExperience'], y=df['salary'], palette='viridis')
plt.xlabel('工作经验 (编码值)', fontweight='bold')
plt.ylabel('薪资 (元)', fontweight='bold')
plt.title('工作经验与薪资分布关系', fontsize=14, fontweight='bold')

# 添加经验编码映射说明
exp_labels = ['0:经验不限', '0.5:在校/应届', '1:1年以内', '2:1-3年', '4:3-5年', '7.5:5-10年', '10:10年以上']
plt.xticks(ticks=range(len(exp_labels)), labels=exp_labels, rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('xgb_experience_salary.png')
plt.close()

# 5. 部分依赖图 (TOP 2 数值型重要特征)
top_numeric_features = feature_importance[feature_importance['Feature'].isin(numerical_features)].head(2)
top_features = top_numeric_features['Feature'].tolist()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for i, feature in enumerate(top_features):
    PartialDependenceDisplay.from_estimator(
        best_model, X_train, features=[feature],
        ax=ax[i], line_kw={"color": "darkred", "lw": 3}
    )
    ax[i].set_title(f'{feature} 部分依赖图', fontweight='bold')
    ax[i].set_ylabel('薪资预测值', fontweight='bold')
    ax[i].grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('xgb_partial_dependence.png')
plt.close()

# 6. 教育程度与薪资关系图
plt.figure()
edu_order = sorted(df['educational'].unique())
edu_labels = {1: '不限', 2: '大专', 3: '本科', 4: '硕士', 5: '博士'}
# 修改第281行
sns.boxplot(x=df['educational'], y=df['salary'], hue=df['educational'], order=edu_order, palette='mako', legend=False)

plt.xlabel('教育程度', fontweight='bold')
plt.ylabel('薪资 (元)', fontweight='bold')
plt.title('教育程度与薪资分布关系', fontsize=14, fontweight='bold')
plt.xticks(ticks=range(len(edu_labels)), labels=[edu_labels[x] for x in edu_order])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('xgb_education_salary.png')
plt.close()

# 7. 模型参数优化热力图 (学习率 vs 树的数量)
param_results = pd.DataFrame(grid_search.cv_results_)
n_estimators = param_grid['regressor__n_estimators']
learning_rates = param_grid['regressor__learning_rate']

# 创建热力图数据
heatmap_data = pd.DataFrame(index=learning_rates, columns=n_estimators)
for i, params in enumerate(param_results['params']):
    n = params['regressor__n_estimators']
    lr = params['regressor__learning_rate']
    score = param_results.loc[i, 'mean_test_score']
    heatmap_data.loc[lr, n] = score

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data.astype(float), annot=True, fmt=".4f", cmap="YlGnBu",
            cbar_kws={'label': 'R² Score'})
plt.xlabel('树的数量 (n_estimators)', fontweight='bold')
plt.ylabel('学习率 (learning_rate)', fontweight='bold')
plt.title('XGBoost参数优化热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('xgb_hyperparameter_heatmap.png')
plt.close()

print("\n所有图表已保存为PNG文件:")
print("1. xgb_feature_importance.png - 特征重要性分布")
print("2. xgb_actual_vs_predicted.png - 实际vs预测薪资散点图")
print("3. xgb_residual_distribution.png - 残差分布图")
print("4. xgb_experience_salary.png - 工作经验与薪资关系")
print("5. xgb_partial_dependence.png - 部分依赖图")
print("6. xgb_education_salary.png - 教育程度与薪资关系")
print("7. xgb_hyperparameter_heatmap.png - 参数优化热力图")