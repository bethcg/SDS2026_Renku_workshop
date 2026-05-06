library(tidyverse)
library(caret)

# 1. Load the data created by the Python script
data <- read_csv("data/health/heart_disease.csv")

# 2. Basic Preprocessing
# Target 'num' is the diameter narrowing (0 = healthy, 1-4 = disease)
# Convert to binary classification: 0 (No) vs 1 (Yes)
data <- data %>%
  mutate(target = ifelse(num > 0, "Disease", "Healthy")) %>%
  mutate(target = as.factor(target)) %>%
  select(-num) # Remove original target

# 3. Simple Logistic Regression Model
print("Training diagnostic model...")
model <- train(target ~ age + sex + cp + trestbps + chol, 
               data = data, 
               method = "glm", 
               family = "binomial",
               na.action = na.omit)

# 4. Save Results
dir.create("outputs/health", recursive = TRUE, showWarnings = FALSE)
sink("outputs/health/model_summary.txt")
print(summary(model))
sink()

# Save a Variable Importance plot
png("outputs/health/feature_importance.png")
plot(varImp(model), main="Key Diagnostic Factors")
dev.off()

print("Analysis complete. Results saved to outputs/health/")
