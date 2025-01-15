#!/opt/software/R/bin/Rscript --vanilla

# Script to calculate mortality score based on the metabolomics data for VOILA project
rm(list = ls())
.libPaths("~/ukb/Scripts_Lieke/Library")
library("haven")
library("dplyr")
library("tidyverse")
library("devtools")
#devtools::install_github("DanieleBizzarri/MetaboRiSc")

r0<-read.csv("round0ratios.csv")
#############transform metabolites################
data <- function(wave) {
  switch(wave,
         "0" = r0,
         "1" = r1,
         "both" = {
           r0$eid <- paste0(r0$eid, "_0.0")
           r1$eid <- paste0(r1$eid, "_1.0")
           full_join(r0, r1)
         },
         "1_un" = run,
         "both_un" = {
           r0$eid <- paste0(r0$eid, "_0.0")
           r1$eid <- paste0(r1$eid, "_1.0")
           full_join(r0, r1)
         },
         stop("Invalid wave parameter.")
  )
}


library("MiMIR")
#MiMIR::startApp()

load("mort_betas.rda")
source('functions_metaboAge2.R')
metabo_names_translator<-readRDS("metabolomic_feature_translator_BBMRI.rds")

for(j in c("0")){
  a <- data(j)
  colnames(a) <- tolower(colnames(a))
 
  a <- a %>% 
    mutate(across(!eid, ~ as.numeric(as.character(.))))
   rownames(a) <-a[,"eid"]
   
   metabo_mat <- a
   ################################
   ## Find the metabolic features names##
   ################################
   #avoid case-sensitive alternative names
   colnames(metabo_mat)<-tolower(colnames(metabo_mat))
   #Looking for alternative names
   nam<-find_BBMRI_names(colnames(metabo_mat))
   i<-which(nam$BBMRI_names %in% metabo_names_translator$BBMRI_names)
   metabo_mat<-metabo_mat[,i]
   colnames(metabo_mat)<-nam$BBMRI_names[i]
  
  
  MH <- MiMIR::comp.mort_score(metabo_mat)
  hist(MH$mortScore)
  if(j == "both" | j == "both_un"){
    MH$eid <- rownames(MH)
    MH$round=substr(MH$eid, nchar(MH$eid)-3, nchar(MH$eid)-2)
    MH$eid <- substr(MH$eid, 1, nchar(MH$eid)-4)
  } else{
    MH$eid <- as.integer(rownames(MH))
  }
  
  MH = select(MH, c(eid, mortScore))
  data.table::fwrite(MH, file=paste0("Round",j,"_MiMIR_MetaboHealth.csv"))
  
  
}

