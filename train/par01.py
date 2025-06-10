import pandas as pd
import numpy as np
import re
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import joblib

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

# 7. 模型构建与调参
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1
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

importances = best_model.named_steps['regressor'].feature_importances_
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n特征重要性TOP 10:")
print(feature_importance.head(10))

# 10. 保存模型
joblib.dump(best_model, '01/salary_predictor.pkl')

# ======================================================
# 以下为新增的绘图代码
# ======================================================

# 设置论文级别的绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun', 'Times New Roman'],  # 中文字体用宋体，英文用Times New Roman
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
plt.title('TOP 15 特征重要性分布', fontsize=16, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('feature_importance.png')
plt.close()

# 2. 实际薪资 vs 预测薪资散点图
plt.figure()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None, color='royalblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('实际薪资 (元)', fontweight='bold')
plt.ylabel('预测薪资 (元)', fontweight='bold')
plt.title('实际薪资 vs 预测薪资 (R² = {:.4f})'.format(r2), fontsize=14, fontweight='bold')
plt.grid(linestyle='--', alpha=0.5)
plt.text(0.05, 0.95, f'RMSE = {rmse:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.3))
plt.savefig('actual_vs_predicted.png')
plt.close()

# 3. 残差分布图
residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True, color='teal', bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('残差 (实际薪资 - 预测薪资)', fontweight='bold')
plt.ylabel('频数', fontweight='bold')
plt.title('预测残差分布 (均值 = {:.2f}, 标准差 = {:.2f})'.format(residuals.mean(), residuals.std()),
          fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('residual_distribution.png')
plt.close()

# 4. 工作经验与薪资关系图
plt.figure()
sns.boxplot(x=df['workExperience'], y=df['salary'], palette='viridis')
plt.xlabel('工作经验 (编码值)', fontweight='bold')
plt.ylabel('薪资 (元)', fontweight='bold')
plt.title('工作经验与薪资分布关系', fontsize=14, fontweight='bold')

# 添加经验编码映射说明
exp_labels = ['0:经验不限', '0.5:在校/应届', '1:1年以内', '2:1-3年', '4:3-5年', '7.5:5-10年', '10:10年以上']
plt.xticks(ticks=range(len(exp_labels)), labels=exp_labels, rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig('experience_salary.png')
plt.close()

# 5. 部分依赖图 (TOP 2 数值型重要特征)
# 只选择数值型特征
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
plt.savefig('partial_dependence.png')
plt.close()

# 6. 教育程度与薪资关系图
plt.figure()
edu_order = sorted(df['educational'].unique())
edu_labels = {1: '不限', 2: '大专', 3: '本科', 4: '硕士', 5: '博士'}
sns.boxplot(x=df['educational'], y=df['salary'], order=edu_order, palette='mako')
plt.xlabel('教育程度', fontweight='bold')
plt.ylabel('薪资 (元)', fontweight='bold')
plt.title('教育程度与薪资分布关系', fontsize=14, fontweight='bold')
plt.xticks(ticks=range(len(edu_labels)), labels=[edu_labels[x] for x in edu_order])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig('education_salary.png')
plt.close()

# 7. 模型参数优化热力图 (n_estimators vs max_depth)
param_results = pd.DataFrame(grid_search.cv_results_)
n_estimators = param_grid['regressor__n_estimators']
max_depth = param_grid['regressor__max_depth']

# 创建热力图数据
heatmap_data = pd.DataFrame(index=max_depth, columns=n_estimators)
for i, params in enumerate(param_results['params']):
    n = params['regressor__n_estimators']
    d = params['regressor__max_depth'] or 'None'
    score = param_results.loc[i, 'mean_test_score']
    heatmap_data.loc[d, n] = score

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data.astype(float), annot=True, fmt=".4f", cmap="YlGnBu",
            cbar_kws={'label': 'R² Score'})
plt.xlabel('树的数量 (n_estimators)', fontweight='bold')
plt.ylabel('最大深度 (max_depth)', fontweight='bold')
plt.title('网格搜索参数优化热力图', fontsize=14, fontweight='bold')
plt.savefig('hyperparameter_heatmap.png')
plt.close()

print("\n所有图表已保存为PNG文件:")
print("1. feature_importance.png - 特征重要性分布")
print("2. actual_vs_predicted.png - 实际vs预测薪资散点图")
print("3. residual_distribution.png - 残差分布图")
print("4. experience_salary.png - 工作经验与薪资关系")
print("5. partial_dependence.png - 部分依赖图")
print("6. education_salary.png - 教育程度与薪资关系")
print("7. hyperparameter_heatmap.png - 参数优化热力图")