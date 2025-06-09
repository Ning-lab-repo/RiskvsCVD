#####1.sankey---------------
library(tidyverse)
library(ggsankey)
library(ggplot2)
library(cols4all)
library(dittoSeq)
setwd("G:\\CVD\\17.feature")
library(readxl)

df <- read_excel("2.feature.xlsx")
df1 <- read_excel("G:\\CVD\\14.upset\\HTN-SHAP10.xlsx")
df2 <- read_excel("G:\\CVD\\14.upset\\DM-SHAP10.xlsx")
df3 <- read_excel("G:\\CVD\\14.upset\\HCL-SHAP10.xlsx")
df1 <- df1 [,-7]
df2 <- df2 [,-7]
df3 <- df3 [,-7]
head(df1)
library(tidyr)
library(dplyr)

df_long <- df1 %>%
  pivot_longer(cols = everything(),  
               names_to = "Disease",  
               values_to = "Indicator") 

df_long <- df_long %>%
  mutate(Risk = "HTN") 

head(df_long)

df_long2 <- df2 %>%
  pivot_longer(cols = everything(),  
               names_to = "Disease",  
               values_to = "Indicator")  

df_long2 <- df_long2 %>%
  mutate(Risk = "DM")  
head(df_long2)

df_long3 <- df3 %>%
  pivot_longer(cols = everything(),  
               names_to = "Disease",  
               values_to = "Indicator") 

df_long3 <- df_long3 %>%
  mutate(Risk = "HCL")  

head(df_long3)
df_combined <- rbind(df_long, df_long2, df_long3)
head(df_combined)
df_combined <- df_combined[,c(3,1,2)]
head(df_combined)
df_combined <- df_combined[,c(1,3,2)]

library(ggplot2)
library(ggalluvial)
library(dplyr)
data <- df_combined
df4 <- to_lodes_form(data[,1:ncol(data)], 
                     axes = 1:ncol(data),  
                     id = "value")
print(df4)

library(ggplot2)
library(ggalluvial)
library(RColorBrewer)


color_mapping <- c(
  "HTN" = "#436d46", 
  "DM" = "#5ba566", 
  "HCL" = "#adddb4", 
  "Unstable angina" = "#1A74B2",
  "Acute myocardial infarction" = "#FF7F0E",
  "Chronic ischemic heart disease" = "#259D25",
  "Cerebral infarction" = "#D41C1D",
  "Intracerebral hemorrhage" = "#966ABE",
  "Sequelae of cerebrovascular disease" = "#884F44"
)


all_stratum <- unique(df4$stratum)  

unspecified_stratum <- setdiff(all_stratum, names(color_mapping))  


macaron_colors <- colorRampPalette(brewer.pal(9, "Pastel1"))(length(unspecified_stratum))


final_color_mapping <- c(color_mapping, setNames(macaron_colors, unspecified_stratum))


ggplot(df4, aes(x = x, fill = stratum, label = stratum, 
                stratum = stratum, alluvium = value)) +  
  geom_flow(width = 0.3,             
            curve_type = "sine",     
            alpha = 0.5,             
            color = "white",         
            linewidth = 0.1) +       
  geom_stratum(width = 0.28) +       
  geom_text(stat = "stratum", size = 2, color = "black") +  
  scale_fill_manual(values = final_color_mapping) +  
  theme_void() +                     
  theme(legend.position = "none")  
setwd("G:\\CVD\\19")

ggsave("sankey_plot2.pdf", width = 22, height = 15, dpi = 300)
ggsave("sankey_plot2.png", width = 22, height = 15, dpi = 300)
######2.ridege------------
library(readxl)
library(dplyr)
file1 <- read_excel("G:/CVD/HTN.xlsx")
file2 <- read_excel("G:/CVD/DM.xlsx")
file3 <- read_excel("G:/CVD/HCL.xlsx")
file4 <- read_excel("G:/CVD/UA.xlsx")
file5 <- read_excel("G:/CVD/AMI.xlsx")
file6 <- read_excel("G:/CVD/CHI.xlsx")
file7 <- read_excel("G:/CVD/CI.xlsx")
file8 <- read_excel("G:/CVD/HI.xlsx")
file9 <- read_excel("G:/CVD/SQ.xlsx")
merged_data <- bind_rows(file1, file2, file3, file4, file5, file6, file7, file8, file9)
data <- merged_data
data <-data[,-c(2,60:64)]
library(ggplot2)
library(ggridges)  
data$class <- as.factor(data$class) 

ggplot(data, aes(x = CKMB, y = class, fill = class)) + 
  geom_density_ridges(scale = 4) + 
  scale_fill_cyclical(values = c( "#884F44", "#966ABE", "#D41C1D", "#259D25",
                                 "#FF7F0E","#1A74B2","#adddb4","#5ba566","#436d46"))+ 
  theme(
    panel.background = element_rect(fill = "white", colour = NA), 
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    panel.border = element_blank(),  
    axis.title.y = element_blank(), 
    axis.text.y = element_blank(),  
    axis.ticks.y = element_blank(),  
    axis.text.x = element_text(size = 15,margin = margin(t = -3) ),
    axis.ticks.length = unit(0, "cm"), 
    axis.title.x = element_text(size = 25) 
  )

data2 <- data

for (i in 2:58) {
 
  quantiles <- quantile(data2[[i]], probs = c(0.025, 0.975), na.rm = TRUE)
  
  
  data2[[i]][data2[[i]] < quantiles[1] | data2[[i]] > quantiles[2]] <- NA
}

ggplot(data2, aes(x = CKMB, y = class, fill = class)) + 
  geom_density_ridges(scale = 4) + 
  scale_fill_cyclical(values = c( "#884F44", "#966ABE", "#D41C1D", "#259D25",
                                  "#FF7F0E","#1A74B2","#adddb4","#5ba566","#436d46"))+ 
  theme(
    panel.background = element_rect(fill = "white", colour = NA), 
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    panel.border = element_blank(),  
    axis.title.y = element_blank(),  
    axis.text.y = element_blank(),  
    axis.ticks.y = element_blank(),  
    axis.text.x = element_text(size = 15,margin = margin(t = -3) ), 
    axis.ticks.length = unit(0, "cm"), 
    axis.title.x = element_text(size = 25) 
  )

library(ggplot2)
library(ggridges)
library(tidyr)  
library(dplyr)

#
library(tidyr) 

data_long <- data2 %>%
  pivot_longer(cols = c(CKMB, GLU, CRE, DBIL), 
               names_to = "variable", 
               values_to = "value")

ggplot(data_long, aes(x = value, y = class, fill = class)) + 
  geom_density_ridges(scale = 4) + 
  scale_fill_cyclical(values = c( "#884F44", "#966ABE", "#D41C1D", "#259D25",
                                  "#FF7F0E","#1A74B2","#adddb4","#5ba566","#436d46")) + 
  theme(
    panel.background = element_rect(fill = "white", colour = NA), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(), 
    panel.border = element_blank(),  
    axis.title.y = element_blank(),  
    axis.text.y = element_blank(),  
    axis.ticks.y = element_blank(),  
    axis.text.x = element_text(size = 15, margin = margin(t = -3)), 
    axis.ticks.length = unit(0.25, "cm"), 
    axis.title.x = element_blank()  
  ) +
  facet_wrap(~ variable, scales = "free_x", ncol = 4)  

setwd("G:\\CVD\\20.RIDEGE")
ggsave("RIDEGE.pdf", width = 16, height = 10, dpi = 300)
ggsave("RIDEGE.png", width = 16, height =10, dpi = 300)

#######44444-------------
class_colors <- c(
  "9" = "#884F44", 
  "8" = "#966ABE", 
  "7" = "#D41C1D", 
  "6" = "#259D25",
  "5" = "#FF7F0E",
  "4" = "#1A74B2",
  "3" = "#adddb4",
  "2" = "#5ba566",
  "1" = "#436d46"
)

ggplot(data_long, aes(x = value, y = class, fill = class)) + 
  geom_density_ridges(scale = 4) + 
  scale_fill_manual(values = class_colors) +  
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = 15, margin = margin(t = -3)),
    axis.ticks.length = unit(0.25, "cm"),
    axis.title.x = element_blank(),
    legend.position = "none" 
  ) +
  facet_wrap(~ variable, scales = "free_x", ncol = 4)


setwd("G:\\CVD\\20.RIDEGE")
ggsave("RIDEGE2.pdf", width = 16, height = 10, dpi = 300)
ggsave("RIDEGE2.png", width = 16, height =10, dpi = 300)


