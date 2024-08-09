library("missMDA")
library(tidyverse)
####### Dataset ######
load_raw_data <- function(dataset) {
  if (dataset == "abalone") {
    true_data = read.csv('./true_data/abalone.csv')
    categorical_features <- c("Sex", "Rings")
    integer_features <- c()
  } else if (dataset == "banknote") {
    true_data = read.csv('./true_data/banknote.csv')
    categorical_features <- c('class')
    integer_features <- c()
    
  } else if (dataset == "breast") {
    true_data = read.csv('./true_data/breast.csv')
    categorical_features <- c("Diagnosis")
    integer_features <- c()
    
  } else if (dataset == "concrete") {
    true_data = read.csv('./true_data/concrete.csv')
    categorical_features <- c("Age")
    integer_features <- c()
    
  } else if (dataset == "redwine") {
    true_data = read.csv('./true_data/redwine.csv')
    categorical_features <- c("quality")
    integer_features <- c()
    
  } else if (dataset == "whitewine") {
    true_data = read.csv('./true_data/whitewine.csv')
    categorical_features <- c("quality")
    integer_features <- c()
  } else {
    stop("Dataset not recognized.")
  }
  
  return(list(true_data = true_data, categorical_features = categorical_features, integer_features = integer_features))
}

####### Imputation #########
datasets <- c("abalone", "banknote", "breast", "redwine", "whitewine")
missing_rate <- "0.3"
missing_types <- c("MAR", "MCAR", "MNARQ", "MNARL")
seeds <- c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

result = data.frame()
for (dataset in datasets) {
  for (missing_type in missing_types) {
    print(sprintf('%s %s %s is start!!', dataset, missing_type, missing_rate))
    for (seed in seeds) {
      file_path <- sprintf("./missing_data/%s/%s_%s_%s_%s.csv", dataset, dataset, missing_type, missing_rate, seed)
      data <- read.csv(file_path)
      
      data_list <- load_raw_data(dataset=dataset)
      categorical_features = data_list$categorical_features
      integer_features = data_list$integer_features
      
      true_data = data_list$true_data %>% select(-any_of(categorical_features))
      true <- apply(true_data, 2, function(x) mean(x > mean(x)))

      M = 100
      
      for (col_name in categorical_features) {
        data[[col_name]] <- as.factor(data[[col_name]])
      }
      try({
        imputed_data <- MIFAMD(data, ncp = 1, nboot=M, threshold = 1.0, seed = 0)
      
        for(impu_data in imputed_data$res.MI){
          impu_data[integer_features] = as.data.frame(lapply(impu_data[integer_features], as.integer))
        }
        
        output_file_path <- sprintf(
          "./missMDA/imputed_data/%s/%s_%s_0.3_%s.csv", dataset, dataset, missing_type, seed
        )
        output_dir <- dirname(output_file_path)
        
        if (!dir.exists(output_dir)) {
          dir.create(output_dir, recursive = TRUE)
        }
        write.csv(imputed_data$res.MI, output_file_path, row.names=FALSE)
        
        est <- data.frame()
        var <- data.frame()
        for(impu_data in imputed_data$res.MI){
          impu_data = impu_data %>% select(-any_of(categorical_features))
          
          p <- apply(impu_data, 2, function(x) mean(x > mean(x)))
          est <- rbind(est, p)
          var <- rbind(var, p * (1 - p) / nrow(impu_data))
        }
        colnames(est) = colnames(impu_data)
        colnames(var) = colnames(impu_data)
        
        Q <- colMeans(est)
        U <- colMeans(var) + (M + 1) / M * apply(est, 2, var)
        
        lower <- Q - 1.96 * sqrt(U)
        upper <- Q + 1.96 * sqrt(U)
        
        bias <- mean(abs(Q - true))
        coverage <- mean((lower < true) & (true < upper))
        interval <- mean(upper - lower)
        
        result = rbind(result, data.frame(dataset=dataset, missing_type=missing_type,seed=seed, bias=bias, coverage=coverage, interval=interval))
        write.csv(result, sprintf("./missMDA/imputed_data/%s_%s_%s_%s.csv", dataset, missing_type, missing_rate, seed), row.names=FALSE)
        })  
    }
    print(sprintf('%s %s %s is end!!', dataset, missing_type, missing_rate))
    write.csv(result, sprintf("./missMDA/imputed_data/%s_%s_%s.csv", dataset, missing_type, missing_rate), row.names=FALSE)
  }
}
write.csv(result, "./missMDA/imputed_data/result.csv", row.names=FALSE)