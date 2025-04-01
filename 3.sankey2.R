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

