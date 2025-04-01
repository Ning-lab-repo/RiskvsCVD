import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 文件路径
input_path = r"G:\CVD\3.KNN\总总-去重20836.xlsx"
data = pd.read_excel(input_path)

# 数据预处理：确保报告日期是日期格式（可选）
data['报告日期'] = pd.to_datetime(data['报告日期'], errors='coerce')

# 对第3列到第59列进行标准化
scaler = StandardScaler()
X = data.iloc[:, 2:59]  # 第 3 列到第 59 列为特征
X_standardized = scaler.fit_transform(X)  # 标准化处理
y = data.iloc[:, -1]    # 最后一列为目标变量

# 使用train_test_split按7:3划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42, stratify=y)

# 计算类别权重（适用于类别不平衡问题）
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 定义模型及超参数
rf_params = {'criterion': 'entropy', 
             'max_depth': 8, 
             'min_samples_leaf': 2, 
             'min_samples_split': 2, 
             'n_estimators': 50}
svm_params = {'C': 1.0, 
              'gamma': 'auto', 
              'kernel': 'rbf'}
lr_params = {'C': 0.1, 
             'max_iter': 200, 
             'penalty': 'l2', 
             'solver': 'liblinear'}
xgb_params = {
    'alpha': 1.5,  # 适度降低 L1 约束，避免过度筛选特征
    'lambda': 1.5,  # 适度增加 L2 正则，防止过拟合
    'gamma': 0.2,  # 适度剪枝，避免过度分裂
    'colsample_bytree': 0.75,  # 每棵树看到更多特征
    'subsample': 0.85,  # 让树看到更多数据
    'learning_rate': 0.007,  # 稍微提升学习率，增加拟合能力
    'max_depth': 9,  # 适当减少树的深度，避免过拟合
    'n_estimators': 5200,  # 增加树的数量，弥补低学习率
    'min_child_weight': 3,  # 增加敏感度，减少过拟合
}

# 初始化模型
rf = RandomForestClassifier(**rf_params, random_state=42, class_weight='balanced')
svm = SVC(**svm_params, probability=True, random_state=42, class_weight='balanced')
lr = LogisticRegression(**lr_params, random_state=42, class_weight='balanced')
xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=class_weight_dict[0] / class_weight_dict[1])

# 初始化交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每个模型的交叉验证AUC
cv_auc_rf = []
cv_auc_svm = []
cv_auc_lr = []
cv_auc_xgb = []

# 在训练集上进行5折交叉验证
for train_idx, val_idx in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]  # 使用 .iloc[] 进行位置索引

    # 训练模型
    rf.fit(X_train_fold, y_train_fold)
    svm.fit(X_train_fold, y_train_fold)
    lr.fit(X_train_fold, y_train_fold)
    xgb_model.fit(X_train_fold, y_train_fold)

    # 预测验证集的概率
    y_pred_rf_proba = rf.predict_proba(X_val_fold)[:, 1]
    y_pred_svm_proba = svm.predict_proba(X_val_fold)[:, 1]
    y_pred_lr_proba = lr.predict_proba(X_val_fold)[:, 1]
    y_pred_xgb_proba = xgb_model.predict_proba(X_val_fold)[:, 1]

    # 计算AUC并存储
    cv_auc_rf.append(roc_auc_score(y_val_fold, y_pred_rf_proba))
    cv_auc_svm.append(roc_auc_score(y_val_fold, y_pred_svm_proba))
    cv_auc_lr.append(roc_auc_score(y_val_fold, y_pred_lr_proba))
    cv_auc_xgb.append(roc_auc_score(y_val_fold, y_pred_xgb_proba))

# 输出5折交叉验证的平均AUC
print(f"Average AUC for Random Forest: {np.mean(cv_auc_rf):.4f}")
print(f"Average AUC for SVM: {np.mean(cv_auc_svm):.4f}")
print(f"Average AUC for Logistic Regression: {np.mean(cv_auc_lr):.4f}")
print(f"Average AUC for XGBoost: {np.mean(cv_auc_xgb):.4f}")

# 使用全训练集重新训练模型
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 评估模型在测试集上的性能
y_pred_rf_proba = rf.predict_proba(X_test)[:, 1]
y_pred_svm_proba = svm.predict_proba(X_test)[:, 1]
y_pred_lr_proba = lr.predict_proba(X_test)[:, 1]
y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# 计算 AUC
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_proba)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm_proba)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr_proba)
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)

# 计算 ROC 曲线数据
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_proba)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))

plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {roc_auc_rf:.4f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.4f})")
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC = {roc_auc_lr:.4f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.4f})")

# 绘制随机猜测（对角线）
plt.plot([0, 1], [0, 1], 'k--')
# 设置图形的标题和标签
plt.xlabel("False Positive Rate", fontsize=25)
plt.ylabel("True Positive Rate", fontsize=25)
plt.title("Blood routine+Biochemical detection", fontsize=25)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(False)

# 保存图像为 PNG 和 PDF
save_path_png = r"G:\CVD\5.ROC\2.总3种模式\4组合ROC.png"  # 修改为实际的路径
save_path_pdf = r"G:\CVD\5.ROC\2.总3种模式\4组合ROC.pdf"  # 修改为实际的路径

plt.savefig(save_path_png, dpi=300, bbox_inches='tight')  # 保存为 PNG
plt.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight')  # 保存为 PDF

print(f"ROC 曲线已保存至: {save_path_png} 和 {save_path_pdf}")

# 显示图形 
plt.show()
