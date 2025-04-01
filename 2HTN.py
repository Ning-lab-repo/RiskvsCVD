import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler



# 3 组文件路径
file_paths_1 = [ 
    r"G:\CVD\10.HTN指标分离\HTN1+不稳定心绞痛_组合.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN2+急性心肌梗死_组合.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN3+慢性缺血性心脏病_组合.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN4+脑梗死_组合.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN5+脑内出血_组合.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN6+脑血管病后遗症_组合.xlsx",
    r"G:\CVD\10.HTN指标分离\2HTN7+总CVD_组合.xlsx"
]

file_paths_2 = [ 
    r"G:\CVD\10.HTN指标分离\HTN1+不稳定心绞痛_生化.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN2+急性心肌梗死_生化.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN3+慢性缺血性心脏病_生化.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN4+脑梗死_生化.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN5+脑内出血_生化.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN6+脑血管病后遗症_生化.xlsx",
    r"G:\CVD\10.HTN指标分离\2HTN7+总CVD_生化.xlsx"
]

file_paths_3 = [ 
    r"G:\CVD\10.HTN指标分离\HTN1+不稳定心绞痛_血常规.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN2+急性心肌梗死_血常规.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN3+慢性缺血性心脏病_血常规.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN4+脑梗死_血常规.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN5+脑内出血_血常规.xlsx",
    r"G:\CVD\10.HTN指标分离\HTN6+脑血管病后遗症_血常规.xlsx",
    r"G:\CVD\10.HTN指标分离\2HTN7+总CVD_血常规.xlsx"
]

# 特征范围
feature_ranges = {
    "组合": (3, 59),
    "生化": (3, 35),
    "血常规": (3, 26)
}

# 对应的文件组
file_groups = {
    "组合": file_paths_1,
    "生化": file_paths_2,
    "血常规": file_paths_3
}

# 疾病名称
disease_labels = [
    "Unstable angina", "Acute myocardial infarction", "Chronic ischemic heart disease",
    "Cerebral infarction", "Intracerebral hemorrhage", "Sequelae of cerebrovascular disease", "CVD patients"
]

# 创建子图: 7 行（疾病）× 3 列（文件组）
fig, axs = plt.subplots(7, 3, figsize=(18, 8), constrained_layout=True)

# 交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

# 遍历 3 组文件
for col_idx, (group_name, file_paths) in enumerate(file_groups.items()):
    start_col, end_col = feature_ranges[group_name]  # 该组文件的特征列范围

    # 遍历 7 个疾病
    for row_idx, file_path in enumerate(file_paths):
        ax = axs[row_idx, col_idx]  # 选择对应子图

        # 读取 Excel 数据
        data = pd.read_excel(file_path)

        # 目标变量转换（1-3 为 0，其他为 1）
        y = data.iloc[:, -1].apply(lambda x: 0 if x in [1, 2, 3] else 1)

        # 选择特征
        features = data.iloc[:, start_col:end_col]

        # 数据标准化
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42, stratify=y)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 定义XGBoost模型及其超参数
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

        # 初始化XGB模型
        model = xgb.XGBClassifier(
            **xgb_params, 
            random_state=42, 
            use_label_encoder=False,  # 避免警告
            eval_metric='auc',  # 适用于二分类
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)  # 计算类别权重
        )

        # SHAP 计算重要性
        fold_importance = np.zeros(X_train.shape[1])
        for train_index, test_index in kf.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]  # 用 .iloc 来索引

            model.fit(X_train_fold, y_train_fold)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test_fold)

            fold_importance += np.abs(shap_values.values).mean(axis=0)

        # 计算平均 SHAP 重要性
        fold_importance /= 5
        importance_df = pd.DataFrame({'Feature': features.columns.tolist(), 'SHAP Importance': fold_importance})
        top_features = importance_df.nlargest(10, 'SHAP Importance')

        # 保存为 Excel 文件
        excel_save_path = f"G:/CVD/13.气泡图/{group_name}_{disease_labels[row_idx]}_HTN-importance.xlsx"
        importance_df.to_excel(excel_save_path, index=False)
        top_features.to_excel(f"G:/CVD/13.气泡图/{group_name}_{disease_labels[row_idx]}_HTN-top_features.xlsx", index=False)

        # 绘制气泡图
        ax.scatter(range(len(top_features)), np.zeros(len(top_features)), s=top_features['SHAP Importance'] * 1000, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.set_yticks([])
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels(top_features['Feature'], rotation=45, ha='right', fontsize=12)

# 保存 & 显示图像
plt.savefig(r"G:\CVD\13.气泡图\HTN_合并2.png", format='png', bbox_inches='tight', dpi=300)
plt.savefig(r"G:\CVD\13.气泡图\HTN_合并2.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()
