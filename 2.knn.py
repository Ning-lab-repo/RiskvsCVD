import pandas as pd
from sklearn.impute import KNNImputer


input_path = r"G:\CVD\57_6-CVD+3risk-unique.xlsx"
output_path = r"G:\CVD\3.KNN\all-unique-20836.xlsx"


data = pd.read_excel(input_path)


print("str:", data.shape)
print("colnames：", data.columns)

columns_to_fill = data.columns[2:59]
imputer = KNNImputer(n_neighbors=5) 


data[columns_to_fill] = imputer.fit_transform(data[columns_to_fill])



data.to_excel(output_path, index=False)

print(f"save：{output_path}")
