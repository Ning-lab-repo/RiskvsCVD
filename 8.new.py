import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier  # 导入 XGBoost
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 标签映射
label_mapping = {
    "不稳定性心绞痛": "Unstable angina",
    "急性心肌梗死": "Acute myocardial infarction",
    "慢性缺血性心脏病": "Chronic ischemic heart disease",
    "脑梗死": "Cerebral infarction",
    "脑内出血": "Intracerebral hemorrhage",
    "脑血管病后遗症": "Sequelae of cerebrovascular disease"
}

# 文件路径
input_paths = [
    r"G:\CVD\3.KNN\风险人群+不稳定性心绞痛.xlsx",
    r"G:\CVD\3.KNN\风险人群+急性心肌梗死.xlsx",
    r"G:\CVD\3.KNN\风险人群+慢性缺血性心脏病.xlsx",
    r"G:\CVD\3.KNN\风险人群+脑梗死.xlsx",
    r"G:\CVD\3.KNN\风险人群+脑内出血.xlsx",
    r"G:\CVD\3.KNN\风险人群+脑血管病后遗症.xlsx",
]

# 创建一个图形对象，所有曲线将在同一图上绘制
plt.figure(figsize=(10, 8))

# 遍历文件路径进行处理
for input_path in input_paths:
    data = pd.read_excel(input_path)

    # 数据预处理：确保报告日期是日期格式（可选）
    data['报告日期'] = pd.to_datetime(data['报告日期'], errors='coerce')

    # 对第3列到第59列进行标准化
    scaler = StandardScaler()
    X = data.iloc[:, 2:59]  # 第 3 列到第 59 列为特征
    X_standardized = scaler.fit_transform(X)  # 标准化处理

    # 将目标变量1-3改为0，其他改为1
    y = data.iloc[:, -1]    # 最后一列为目标变量
    y = y.apply(lambda x: 0 if x in [1, 2, 3] else 1)

    # 使用train_test_split按7:3划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, 
                                                        random_state=42, stratify=y)

    # 计算类别权重（适用于类别不平衡问题）
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # 定义XGBoost模型
    xgb_model = XGBClassifier(
        alpha=1.5,  # 适度降低 L1 约束，避免过度筛选特征
        reg_lambda=1.5,  # 适度增加 L2 正则，防止过拟合
        gamma=0.2,  # 适度剪枝，避免过度分裂
        colsample_bytree=0.75,  # 每棵树看到更多特征
        subsample=0.85,  # 让树看到更多数据
        learning_rate=0.007,  # 稍微提升学习率，增加拟合能力
        max_depth=9,  # 适当减少树的深度，避免过拟合
        n_estimators=5200,  # 增加树的数量，弥补低学习率
        min_child_weight=3,  # 增加敏感度，减少过拟合
    )

    # 使用交叉验证评估模型性能
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉验证
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc')

    # 输出交叉验证的AUC结果
    print(f"交叉验证 AUC 得分：{cv_scores}")
    print(f"平均 AUC 得分：{cv_scores.mean():.4f}")

    # 训练模型
    xgb_model.fit(X_train, y_train)

    # 评估模型在测试集上的性能
    y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]  # 获取正类的概率

    # 计算 AUC
    roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)

    # 计算 ROC 曲线数据
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)

    # 提取文件名中的标签部分，并映射到英文
    label = input_path.split('/')[-1].replace('.xlsx', '').split('+')[1]
    label = label_mapping.get(label, label)  # 如果没有找到对应的映射，保持原始标签

    # 绘制 ROC 曲线
    plt.plot(fpr_xgb, tpr_xgb, label=f"Risk Populations vs {label} (AUC = {roc_auc_xgb:.4f})")

# 绘制随机猜测（对角线）
plt.plot([0, 1], [0, 1], 'k--')

# 设置图形的标题和标签
plt.xlabel("False Positive Rate", fontsize=25)
plt.ylabel("True Positive Rate", fontsize=25)
plt.title("Risk Populations vs CVD Patients", fontsize=25)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(False)

# 保存图像为PNG
save_path_png = r"G:\CVD\5.ROC\3.总多分类\8.Risk Populations vs CVD Patients7.png"
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')  # 确保保存图像时不裁剪内容
print(f"ROC 曲线已保存至 PNG: {save_path_png}")

# 保存图像为PDF
save_path_pdf = r"G:\CVD\5.ROC\3.总多分类\8.Risk Populations vs CVD Patients7.pdf"
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')  # 保存为PDF
print(f"ROC 曲线已保存至 PDF: {save_path_pdf}")

# 显示图形 
plt.show()
