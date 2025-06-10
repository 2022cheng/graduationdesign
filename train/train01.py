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

# 10. 保存模型 (可选)
import joblib

joblib.dump(best_model, '01/salary_predictor.pkl')