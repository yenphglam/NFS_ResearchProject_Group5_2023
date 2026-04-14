# SoCal DS Group 5
# Alzheimer's 2 RFs

# Yvette Vargas
# yvargas10
# July 19, 2024
############################################################
# USES ROC TO MEASURE ACCURACY
# RF 1 classifies between HC and Not HC (ROC_AUC 0.921)
# RF 2 classifies between MCI and AD (ROC_AUC 0.901)
############################################################
library(dplyr)
library(ggplot2)
library(stringr)
library(tidyverse) 
library(tidymodels) #Modeling
library(ranger)
library(probably)
library(vip)
library(yardstick)
library(rpart.plot)

## Load in Data
mri <- read.csv("mri.csv")
mriraw <- read.csv("investigator_mri_nacc57.csv")
uds <- read.csv("/Users/yvargas/SoCalDataScience/socalDS-group-5/NACC_data/data_cleaned/uds.csv")

## Data Wrangling ***with group
udsselect <- uds %>% select(
  SEX,
  EDUC,
  NACCAGE,
  VEG,
  ANIMALS,
  TRAILA,
  TRAILB,
  CRAFTDRE,
  MINTTOTS,
  DIGBACCT,
  MEMPROB,
  DROPACT,
  WRTHLESS,
  BETTER,
  BORED,
  HELPLESS,
  TAXES,
  BILLS,
  REMDATES,
  TRAVEL
)


# WRANGLING UDSD TO COGSTAT
cog_status <- factor(uds$NACCUDSD,
                     levels = c(1, 2, 3, 4),
                     labels = c("HC", "Impaired", "MCI", "AD"))
cog_hc_or_not <- factor(uds$NACCUDSD,
                      levels = c(1, 2, 3, 4),
                      labels = c("HC", "NHC", "NHC", "NHC"))

# complete cases of MCI and AD subject
factor_vars <- c("TAXES", "BILLS", "REMDATES", "TRAVEL", "MEMPROB", 
                 "DROPACT", "WRTHLESS", "BETTER", "BORED", "HELPLESS", "SEX")


##########################################################################################
# RF 1: HC vs Not HC
##########################################################################################
uds_cog_stat <- udsselect %>%
  # Factor multiple variables using across
  mutate(across(all_of(factor_vars), factor)) %>%
  # Factor the cog_status variable separately
  # mutate(cog_status = factor(cog_status)) %>% # commented out because otherwise this variable is used solely in the tree
  mutate(cog_hc_or_not = factor(cog_hc_or_not)) %>%
  na.omit()


# Training and Test Set
set.seed(777)
udscog_split <- initial_split(uds_cog_stat, prop = 0.8)
udscog_train <- training(udscog_split)
udscog_test <- testing(udscog_split)


########################
# Classification Trees
########################

# tree-tidy model
# treeR_model <- decision_tree(mode = "regression", engine = "rpart",
#                            cost_complexity = tune())
# let's just tune the cost-complexity parameter
# can be used for CDRSUM AND MMSE below this chunk


# treeC-tidy model
#treeC_model <- set_mode(treeR_model, "classification")

# Equivalent to:
treeC_model <- decision_tree(mode = "classification", engine = "rpart", cost_complexity = tune())
# because we are still using rpart and tuning the cost-complexity parameter


# treeC-tidy recipe AD}
treeC_recipe_AD <- recipe(cog_hc_or_not ~ . ,data = udscog_train)

treeC_wflow_AD <- workflow() |>
  add_model(treeC_model) |>
  add_recipe(treeC_recipe_AD)


# tune Cmodel kfold 1 AD
set.seed(777)
AD_kfold <- vfold_cv(udscog_train, v = 5, repeats = 3) 

# changed mn_log_loss to auc_roc
treeC_tune2 <- tune_grid(treeC_model, 
                         treeC_recipe_AD, 
                         resamples = AD_kfold, 
                         metrics = metric_set(roc_auc),
                         grid = grid_regular(cost_complexity(range = c(-2, 0)), levels = 10))

autoplot(treeC_tune2)

# changed mn_log_loss to roc_auc
treeC_best2 <- select_by_one_std_err(
  treeC_tune2,
  metric = "roc_auc",
  desc(cost_complexity)
)
treeC_best2


# fit treeC an actual tiny tree
treeC_wflow2_final <- finalize_workflow(treeC_wflow_AD, parameters = treeC_best2) 

treeC_fit2 <- fit(treeC_wflow2_final, data = udscog_train)

rpart.plot(extract_fit_engine(treeC_fit2), main = "Decision Tree Plot", type = 1, extra = 100, under = TRUE, cex = 0.6)


# extract_fit_engine(treeC_fit2) %>%
#   plot()


### Bagging and Random Forests for Classification
# rfC-tidy model
rfC_model <- rand_forest(mode = "classification", engine = "ranger") |>
  set_args(seed = 777,
           importance = "permutation",
           mtry = tune()
  )

rfC_recipe_AD <- recipe(
  cog_hc_or_not ~ .,
  data = udscog_train
)


rfC_wflow_AD <- workflow() |>
  add_model(rfC_model) |>
  add_recipe(rfC_recipe_AD)


# tune model kfold rfC}
# I'm sure there's a better way, but this works
n_predictorsC <- sum(rfC_recipe_AD$var_info$role == "predictor")
manual_gridC <- expand.grid(mtry = seq(1, n_predictorsC))

# changed mn_log_loss to auc_roc
rfC_tune1 <- tune_grid(rfC_model, 
                       rfC_recipe_AD, 
                       resamples = AD_kfold, 
                       metrics = metric_set(roc_auc, accuracy),
                       grid = manual_gridC)

# took 20 minutes to run
autoplot(rfC_tune1)


# select best rfC
# Ted used mn_log_loss, changed to roc_auc
# mn_log_loss if accurate probability extimates are crucial
# roc is model's ability to distinuguish between classes is the focus
rfC_best <- select_best(
  rfC_tune1,
  metric = "roc_auc"
)

# fit rfC-tidy model
rfC_wflow_final <- finalize_workflow(rfC_wflow_AD, parameters = rfC_best) 
rfC_fit <- fit(rfC_wflow_final, data = udscog_train)
rfC_fit


# rfC OOB Brier Score and vip
rfC_engine <- rfC_fit |> extract_fit_engine()
rfC_engine
#extract_fit_engine(treeC_fit2) |>
#plot(
#ylim = c(-0.2, 1.2))
#extract_fit_engine(treeC_fit2) |>
#text(cex = 0.5)

rfC_engine |> pluck("prediction.error")

vip(rfC_engine, main = "Variable importance plot" , scale = TRUE)


# log-loss multiple cross entropy
rfC_multi_AD3 <- broom::augment(rfC_fit, new_data = udscog_test)

# ROC
roc_auc(rfC_multi_AD3,
        truth = cog_hc_or_not,
        .pred_HC)


##########################################################################################
# RF 2: MCI vs AD
##########################################################################################

uds_mci_ad_only <- uds %>%
  select(SEX, EDUC, NACCAGE, VEG, ANIMALS, TRAILA, TRAILB, CRAFTDRE, MINTTOTS, DIGBACCT, MEMPROB,
    DROPACT, WRTHLESS, BETTER, BORED, HELPLESS, TAXES, BILLS, REMDATES, TRAVEL, NACCUDSD) %>%
  filter(NACCUDSD == 3 | NACCUDSD == 4)
  
cog_status_mci_ad_only <- factor(uds_mci_ad_only$NACCUDSD,
       levels = c(3, 4),
       labels = c("MCI", "AD"))

uds_mci_ad_only <- uds_mci_ad_only %>%
  # Factor multiple variables using across
  mutate(across(all_of(factor_vars), factor)) %>%
  # Factor the cog_status variable separately
  mutate(cog_status = factor(cog_status_mci_ad_only)) %>%
  select(-NACCUDSD) %>%
  na.omit()


# Training and Test Set
set.seed(777)
udscog3_split <- initial_split(uds_mci_ad_only, prop = 0.8)
udscog3_train <- training(udscog3_split)
udscog3_test <- testing(udscog3_split)


########################
# Classification Trees
########################

# tree-tidy model
# treeR_model <- decision_tree(mode = "regression", engine = "rpart",
#                            cost_complexity = tune())
# let's just tune the cost-complexity parameter
# can be used for CDRSUM AND MMSE below this chunk


# treeC-tidy model
#treeC_model <- set_mode(treeR_model, "classification")

# Equivalent to:
treeC_model3 <- decision_tree(mode = "classification", engine = "rpart", cost_complexity = tune())
# because we are still using rpart and tuning the cost-complexity parameter


# treeC-tidy recipe AD}
treeC_recipe_AD3 <- recipe(cog_status ~ . ,data = udscog3_train)

treeC_wflow_AD3 <- workflow() |>
  add_model(treeC_model3) |>
  add_recipe(treeC_recipe_AD3)


# tune Cmodel kfold 1 AD
set.seed(777)
AD_kfold3 <- vfold_cv(udscog3_train, v = 5, repeats = 3) 

# changed mn_log_loss to auc_roc
treeC_tune3 <- tune_grid(treeC_model3, 
                         treeC_recipe_AD3, 
                         resamples = AD_kfold3, 
                         metrics = metric_set(roc_auc),
                         grid = grid_regular(cost_complexity(range = c(-3, 0)), levels = 10))

autoplot(treeC_tune3)

# changed mn_log_loss to roc_auc
treeC_best3 <- select_by_one_std_err(
  treeC_tune3,
  metric = "roc_auc",
  desc(cost_complexity)
)
treeC_best3


# fit treeC an actual tiny tree
treeC_wflow3_final <- finalize_workflow(treeC_wflow_AD3, parameters = treeC_best2) 

treeC_fit3 <- fit(treeC_wflow3_final, data = udscog3_train)

rpart.plot(extract_fit_engine(treeC_fit3), main = "Decision Tree Plot", type = 1, extra = 100, under = TRUE, cex = 0.6,
           roundint =FALSE)


# extract_fit_engine(treeC_fit2) %>%
#   plot()


### Bagging and Random Forests for Classification
# rfC-tidy model
rfC_model3 <- rand_forest(mode = "classification", engine = "ranger") |>
  set_args(seed = 777,
           importance = "permutation",
           mtry = tune()
  )

rfC_recipe_AD3 <- recipe(
  cog_status ~ .,
  data = udscog3_train
)


rfC_wflow_AD3 <- workflow() |>
  add_model(rfC_model3) |>
  add_recipe(rfC_recipe_AD3)


# tune model kfold rfC}
# I'm sure there's a better way, but this works
n_predictorsC3 <- sum(rfC_recipe_AD3$var_info$role == "predictor")
manual_gridC3 <- expand.grid(mtry = seq(1, n_predictorsC3))

# changed mn_log_loss to auc_roc
rfC_tune3 <- tune_grid(rfC_model3, 
                       rfC_recipe_AD3, 
                       resamples = AD_kfold3, 
                       metrics = metric_set(roc_auc, accuracy),
                       grid = manual_gridC3)

# took 20 minutes to run
autoplot(rfC_tune3)


# select best rfC
# Ted used mn_log_loss, changed to roc_auc
# mn_log_loss if accurate probability extimates are crucial
# roc is model's ability to distinuguish between classes is the focus
rfC_best <- select_best(
  rfC_tune3,
  metric = "roc_auc"
)

# fit rfC-tidy model
rfC_wflow_final3 <- finalize_workflow(rfC_wflow_AD3, parameters = rfC_best) 
rfC_fit3 <- fit(rfC_wflow_final3, data = udscog3_train)
rfC_fit3


# rfC OOB Brier Score and vip
rfC_engine3 <- rfC_fit3 |> extract_fit_engine()
rfC_engine3
#extract_fit_engine(treeC_fit2) |>
#plot(
#ylim = c(-0.2, 1.2))
#extract_fit_engine(treeC_fit2) |>
#text(cex = 0.5)

rfC_engine3 |> pluck("prediction.error")

vip(rfC_engine3, main = "Variable importance plot" , scale = TRUE)


# log-loss multiple cross entropy
rfC_multi_AD3 <- broom::augment(rfC_fit3, new_data = udscog3_test)

# ROC
roc_auc(rfC_multi_AD3,
        truth = cog_status,
        .pred_MCI)