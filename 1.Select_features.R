
library(openxlsx)
setwd("G:\\CVD")
data <- read.xlsx("6-CVD+3.xlsx")

str(data)
na_counts <- colSums(is.na(data[, 3:61]))  
na_ratios <- na_counts / nrow(data)        
min(na_ratios)
max(na_ratios)

valid_columns <- na_ratios <= 0.5

data_filtered <- data[, c(1:2, which(valid_columns) + 2, (ncol(data)-5):ncol(data))]
colnames(data_filtered)
library(writexl)
setwd("G:\\CVD")
write_xlsx(data_filtered, "1.feature.xlsx")

setwd("G:\\CVD")
data2 <- read.xlsx("3_population.xlsx")

common_columns <- intersect(names(data_filtered), names(data2))

data2 <- data2[, common_columns]

library(readxl)
library(openxlsx)

# 定义所有的文件路径
file_paths <- c(
  "G:/CVD/acute_class.xlsx",
  "G:/CVD/UA_class.xlsx",
  "G:/CVD/CHI_class.xlsx",
  "G:/CVD/CI_class.xlsx",
  "G:/CVD/HI_class.xlsx",
  "G:/CVD/SQ_class.xlsx",
  "G:/CVD/DM.xlsx",
  "G:/CVD/3RISK.xlsx",
  "G:/CVD/3RISK_POPULATION.xlsx",
  "G:/CVD/6-CVD.xlsx",
  "G:/CVD/6-CVD+3RISK.xlsx",
  "G:/CVD/6-CVD-UNIQUE.xlsx",
  "G:/CVD/HTN.xlsx",
  "G:/CVD/HCL.xlsx"
)


output_folder <- "G:\\CVD\\2.57FEATURE"

for (file_path in file_paths) {

  data <- read_excel(file_path)
 
  common_columns <- intersect(names(data), names(data2))  
  filtered_data <- data[, common_columns]  

  output_file <- paste0(output_folder, "/", "57_", basename(file_path))
  write.xlsx(filtered_data, output_file)  
}


