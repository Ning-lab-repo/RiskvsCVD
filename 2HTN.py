import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


file_paths_1 = [ 
    r"G:\CVD\10.HTN\HTN1+heart_all.xlsx",
    r"G:\CVD\10.HTN\HTN2+acute_all.xlsx",
    r"G:\CVD\10.HTN\HTN3+chronic_all.xlsx",
    r"G:\CVD\10.HTN\HTN4+brain_all.xlsx",
    r"G:\CVD\10.HTN\HTN5+Intracerebral_all.xlsx",
    r"G:\CVD\10.HTN\HTN6+Sequelae_all.xlsx",
    r"G:\CVD\10.HTN\2HTN7+CVD_all.xlsx"
]

file_paths_2 = [ 
    r"G:\CVD\10.HTN\HTN1+heart_bio.xlsx",
    r"G:\CVD\10.HTN\HTN2+acute_bio.xlsx",
    r"G:\CVD\10.HTN\HTN3+chronic_bio.xlsx",
    r"G:\CVD\10.HTN\HTN4+brain_bio.xlsx",
    r"G:\CVD\10.HTN\HTN5+Intracerebral_bio.xlsx",
    r"G:\CVD\10.HTN\HTN6+Sequelae_bio.xlsx",
    r"G:\CVD\10.HTN\2HTN7+CVD_bio.xlsx"
]

file_paths_3 = [ 
    r"G:\CVD\10.HTN\HTN1+heart_blood.xlsx",
    r"G:\CVD\10.HTN\HTN2+acute_blood..xlsx",
    r"G:\CVD\10.HTN\HTN3+chronic_blood..xlsx",
    r"G:\CVD\10.HTN\HTN4+brain_blood..xlsx",
    r"G:\CVD\10.HTN\HTN5+Intracerebral_blood..xlsx",
    r"G:\CVD\10.HTN\HTN6+Sequelae_blood..xlsx",
    r"G:\CVD\10.HTN\2HTN7+CVD_blood..xlsx"
]


feature_ranges = {
    "all": (3, 59),
    "bio": (3, 35),
    "blood": (3, 26)
}


file_groups = {
    "all": file_paths_1,
    "bio": file_paths_2,
    "blood": file_paths_3
}


disease_labels = [
    "Unstable angina", "Acute myocardial infarction", "Chronic ischemic heart disease",
    "Cerebral infarction", "Intracerebral hemorrhage", "Sequelae of cerebrovascular disease", "CVD patients"
]

fig, axs = plt.subplots(7, 3, figsize=(18, 8), constrained_layout=True)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

for col_idx, (group_name, file_paths) in enumerate(file_groups.items()):
    start_col, end_col = feature_ranges[group_name]  

  
    for row_idx, file_path in enumerate(file_paths):
        ax = axs[row_idx, col_idx]  

        data = pd.read_excel(file_path)

        y = data.iloc[:, -1].apply(lambda x: 0 if x in [1, 2, 3] else 1)

        features = data.iloc[:, start_col:end_col]

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42, stratify=y)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        xgb_params = {
            'alpha': 1.5,  
            'lambda': 1.5,  
            'gamma': 0.2,  
            'colsample_bytree': 0.75,  
            'subsample': 0.85,  
            'learning_rate': 0.007,  
            'max_depth': 9,  
            'n_estimators': 5200,  
            'min_child_weight': 3,  
        }

       
        model = xgb.XGBClassifier(
            **xgb_params, 
            random_state=42, 
            use_label_encoder=False,  
            eval_metric='auc',  
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)  
        )

      
        fold_importance = np.zeros(X_train.shape[1])
        for train_index, test_index in kf.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]  

            model.fit(X_train_fold, y_train_fold)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test_fold)

            fold_importance += np.abs(shap_values.values).mean(axis=0)

        fold_importance /= 5
        importance_df = pd.DataFrame({'Feature': features.columns.tolist(), 'SHAP Importance': fold_importance})
        top_features = importance_df.nlargest(10, 'SHAP Importance')

   
        excel_save_path = f"G:/CVD/13.bubble/{group_name}_{disease_labels[row_idx]}_HTN-importance.xlsx"
        importance_df.to_excel(excel_save_path, index=False)
        top_features.to_excel(f"G:/CVD/13.bubble/{group_name}_{disease_labels[row_idx]}_HTN-top_features.xlsx", index=False)

     
        ax.scatter(range(len(top_features)), np.zeros(len(top_features)), s=top_features['SHAP Importance'] * 1000, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.set_yticks([])
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels(top_features['Feature'], rotation=45, ha='right', fontsize=12)


plt.savefig(r"G:\CVD\13.bubble\HTN_bubble2.png", format='png', bbox_inches='tight', dpi=300)
plt.savefig(r"G:\CVD\13.bubble\HTN_bubble2.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()
