library(ScottKnottESD)
library(dplyr)

ORGS = c('Eclipse','Libreoffice','Gerrithub','All')
DATA_PATH = 'C:/Users/Motaz/Desktop/work/prepare_tex_tables/CEC2023'
ML_ALGO_LIST = c('NSGA3_LR_recall_TNR_churn_max', 'NSGA3_LR_recall_TNR_churn_median','SOTA_islam')
METRICS = c('popt','recall_at_0.2','auc')

compare_algorithm <- function(orgs,data,metric,algos,greater_is_better = TRUE){
  results <- data.frame(Organization = character(),model = character(),group = double(),mean = double(),median = double())
  for (org_name in orgs) {
    cat('working on org',org_name,'\n')
    org_data = subset(data,org == org_name)
    
    cat('working on metric',metric,'\n')
    
    algos_list = NULL
    for (algo in algos) {
      cat('Working on model',algo,'\n')
      x <- unlist(subset(org_data,model==algo)[metric],use.names = FALSE)
      if (greater_is_better == FALSE){
        x <- x*-1
      }
      algos_list[[algo]] <- x
    }
    metric_df =  as.data.frame(algos_list)
    sk <- sk_esd(metric_df)
    for (algo in algos) {
      x <- unlist(subset(org_data,model==algo)[metric],use.names = FALSE)
      new_entry <- list(Organizatation= org_name,group=sk$groups[[algo]],model=algo,mean= mean(x),median = median(x))
      results <- rbind(results,new_entry)
    }
    print(sk)
    plot(sk)
  }
  return(results)
}
#main 
ALL_ORGS_DATA <- read.csv(paste(DATA_PATH,'/','ready_data_all.csv',sep=''))
ALL_ORGS_DATA_CAT_ALL <- data.frame(ALL_ORGS_DATA)
ALL_ORGS_DATA_CAT_ALL$org <- 'All'

ALL_ORGS_DATA <- rbind(ALL_ORGS_DATA,ALL_ORGS_DATA_CAT_ALL)
results_popt <- compare_algorithm(orgs=ORGS,data=ALL_ORGS_DATA,metric="popt" ,algos=ML_ALGO_LIST,greater_is_better = TRUE)
results_recall_at_02 <- compare_algorithm(orgs=ORGS,data=ALL_ORGS_DATA,metric="recall_at_0.2",algos=ML_ALGO_LIST,greater_is_better = TRUE)
results_auc <- compare_algorithm(orgs=ORGS,data=ALL_ORGS_DATA,metric="auc",algos=ML_ALGO_LIST,greater_is_better = TRUE)
write.csv(results_popt,paste(DATA_PATH,'/','popt_RQ2.csv',sep=''), row.names = FALSE)
write.csv(results_recall_at_02,paste(DATA_PATH,'/','recall_at_02_RQ2.csv',sep=''), row.names = FALSE)
write.csv(results_auc,paste(DATA_PATH,'/','auc_RQ2.csv',sep=''), row.names = FALSE)