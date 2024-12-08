import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 加载数据
file_path = r"D:\预测模型论文\2015脑卒中衰弱预测模型变量(删除缺失值 ID).xlsx"
df = pd.read_excel(file_path)

# 提取输入特征 X 和输出标签 y
X = df.drop(columns=['frailty'])  # 选择所有输入特征，移除 frailty 列
y = df['frailty']  # 选择 frailty 列作为标签

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义随机森林分类器
clf = RandomForestClassifier(n_estimators=33, max_depth=5, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 保存模型
with open('random_forest_model_frailty.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

print("模型已训练并保存为 'random_forest_model_frailty.pkl'")
