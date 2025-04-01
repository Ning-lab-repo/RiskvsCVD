library(UpSetR)
library(openxlsx)
data <- read.xlsx("G:\\CVD\\14.upset\\HTN-SHAP10.xlsx",check.names = FALSE)

colnames(data) <- gsub("\\.", " ", colnames(data))
print(colnames(data))
common_elements <- Reduce(intersect, lapply(data, unique))
setwd("G:\\CVD\\14.upset")
#pdf--------
pdf("upset-HTN2.pdf", width = 8, height = 4.8)  # Save as PDF format, specify width and height in inches (1200x800 pixels at 300 dpi)
# Set the plot parameters, adjust margins
# par(mar = c(5, 5, 2, 2) + 0.1)  # Increase bottom and left margins
upset(fromList(data), 
      nsets = 8,                 
      main.bar.color = "#FC6B17",
      show.numbers = FALSE,
      point.size = 3.5,
      nintersects = 100,
      line.size = 1,
      mainbar.y.label = "Intersection size",
      sets.x.label = " ",
      text.scale = c(2, 2, 2, 1.5, 1.5, 2),
      matrix.color = "#FC6B17"
) 
dev.off()







