library(tidyverse)
library(caret)

# INSTRUCTIONS: modify the data folders so you can read data and write the results.

# 1. Load the data stored in your preferred data storage provider (e.g. Dropbox or Google Drive)
data <- read_csv("~/work/health-data/heart_disease.csv")

# 2. Basic preprocessing
# Target 'num' is the diameter narrowing (0 = healthy, 1-4 = disease)
# Convert to binary classification: 0 (No) vs 1 (Yes)
data <- data %>%
  mutate(target = ifelse(num > 0, "Disease", "Healthy")) %>%
  mutate(target = as.factor(target)) %>%
  select(-num) # Remove original target

# 3. Simple logistic regression model
print("Training diagnostic model...")
model <- train(target ~ age + sex + cp + trestbps + chol, 
               data = data, 
               method = "glm", 
               family = "binomial",
               na.action = na.omit)

# 4. Save results
dir.create("~/work/health-data/outputs/health", recursive = TRUE, showWarnings = FALSE)
sink("~/work/health-data/model_summary.txt")
print(summary(model))
sink()

# Save a variable importance plot
png("~/work/health-data/feature_importance.png")
plot(varImp(model), main="Key Diagnostic Factors")
dev.off()

print("Analysis complete. Results saved to outputs/health/")
