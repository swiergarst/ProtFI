

#Function to translate Nightingale metabolomics alternative metabolite names to the ones used in BBMRI-nl
find_BBMRI_names<-function(names){
  names<-tolower(names)
  new_names <- names %>% purrr::map_chr(function(id) {
    # Look through the alternative_ids
    hits <-
      purrr::map_lgl(
        metabo_names_translator$alternative_names,
        ~ id %in% .
      )
    
    # If one unambiguous hit, return it.
    if (sum(hits) == 1L) {
      return(metabo_names_translator$BBMRI_names[hits])
      # If not found, give a warning and pass through the input.
    } else {
      warning("Biomarker not found: ", id, call. = FALSE)
      return(id)
    } 
  })
  n<-data.frame(uploaded=names,BBMRI_names=new_names)
  return(n)
}

# Quality control of the metabolomics matrix
QCprep<-function(mat,PARAM,quiet=F,Nmax_miss=1,Nmax_zero=1){
  ## mat is the input NH matrix; it may contain a mixture of flags and metabolites in the columns.
  ## PARAM is a list holding the parameters of the pipeline
  # 0. Start:
  if(!quiet){
    cat("|| FILTERING \n")
    cat(report.dim(mat,header="Start"))
  }
  # 1. Subset required metabolites:
  mat <- subset.metabolites.overlap(mat,metabos=PARAM$MET,quiet=quiet)
  # 2. Subset samples on missingness:
  mat <- subset.samples.miss(mat,Nmax=Nmax_miss,quiet=quiet)
  # 3. Subset samples on zeros:
  mat <- subset.samples.zero(mat,Nmax=Nmax_zero,quiet=quiet)
  # 4. Subset samples on SD:
  mat1 <- subset.samples.sd_no_log(as.matrix(mat),MEAN=PARAM_MetaboAge2$BBMRI_mean,
                                  SD=PARAM_MetaboAge2$BBMRI_sd,quiet=quiet)
  sample_names<-rownames(mat)
  # 5. Perform scaling:
  if(!quiet){
    cat("| Performing scaling ... ")
    mat <- sapply(PARAM_MetaboAge2$MET,function(x) (mat[,x]-PARAM_MetaboAge2$BBMRI_mean[x])/PARAM_MetaboAge2$BBMRI_sd[x])
    rownames(mat)<-sample_names
    cat(" DONE!\n")
  }
  # 6. Perform imputation:
  if(!quiet){
    cat("| Imputation ... ")
    mat <- impute.miss(mat)
    cat(" DONE!\n")
  }
  return(mat)
}

apply.fit.metaboAge<-function(mat,PARAM,model_type="LM"){
  if(model_type=="LM"){
    # Resort:
    BETA <- PARAM_MetaboAge2$models_betas[1,colnames(mat)]
    INTC <- PARAM_MetaboAge2$models_betas[1,1]
  }else if(model_type=="EN"){
    # Resort:
    BETA <- PARAM_MetaboAge2$models_betas[2,colnames(mat)]
    INTC <- PARAM_MetaboAge2$models_betas[2,1]
  }
  # Predict age:
  AGE <- data.frame(samp_id=rownames(mat), pred_age=as.vector(mat %*% BETA) + as.vector(INTC),stringsAsFactors=FALSE)
  return(AGE)
}

load.param<-function(pathIN){
  if(!is.null(pathIN) & length(pathIN)==1 & all(is.character(pathIN))){
    if(!file.exists(pathIN)){
      stop("Supplied path does not exist")
    } else {
      return(get(load(pathIN)))
    }
  }
}

subset.samples.sd<-function(x,MEAN,SD,quiet=FALSE){
  MEAN <- MEAN[colnames(x)]
  SD <- SD[colnames(x)]
  Dummi <- log(x)
  Dummi[which(Dummi==-Inf)] <- NA
  # Exclude persons being an outlier:
  outl_samp <- rownames(Dummi)[unique(which(((Dummi > t(replicate(nrow(Dummi),MEAN)) + 5*t(replicate(nrow(Dummi),SD))) | (Dummi < t(replicate(nrow(Dummi),MEAN)) - 5*t(replicate(nrow(Dummi),SD)))),arr.ind=TRUE)[,"row"])]
  sample_names <- setdiff(rownames(Dummi),outl_samp)
  x <- x[sample_names,,drop=FALSE]
  if(!quiet){
    cat(report.dim(x,header=paste0("Pruning samples on 5SD")))
  }
  return(invisible(x))
}

subset.samples.sd_no_log<-function(x,MEAN,SD,quiet=FALSE,d=5){
  MEAN <- MEAN[colnames(x)]
  SD <- SD[colnames(x)]
  Dummi <- x
  #Dummi[which(Dummi==-Inf)] <- NA
  # Exclude persons being an outlier:
  outl_samp <- rownames(Dummi)[unique(which(((Dummi > t(replicate(nrow(Dummi),MEAN)) + d*t(replicate(nrow(Dummi),SD))) | (Dummi < t(replicate(nrow(Dummi),MEAN)) - d*t(replicate(nrow(Dummi),SD)))),arr.ind=TRUE)[,"row"])]
  sample_names <- setdiff(rownames(Dummi),outl_samp)
  x <- x[sample_names,,drop=FALSE]
  if(!quiet){
    cat(report.dim(x,header=paste0("Pruning samples on 5SD")))
  }
  return(invisible(x))
}

report.dim<-function(x,header,trailing="50"){
  return(paste0(sprintf(paste0("%-",trailing,"s"),paste0("| ",header,": ")),sprintf("%4s",ncol(x))," metabolites x ",sprintf("%4s",nrow(x))," samples \n"))
}

subset.metabolites.overlap<-function(x,metabos,quiet=FALSE){
  x <- x[,intersect(colnames(x),metabos),drop=FALSE]
  if(!quiet){
    cat(report.dim(x,header="Selecting metabolites"))
  }
  return(invisible(x))
}

subset.samples.miss<-function(x,Nmax=1,quiet=FALSE){
  MISS <- colSums(is.na(t(x)))
  x <- x[which(MISS<=Nmax),,drop=FALSE]
  if(!quiet){
    cat(report.dim(x,header=paste0("Pruning samples on missing values [Nmax>=",Nmax,"]")))
  }
  return(invisible(x))
}

subset.samples.zero<-function(x,Nmax=1,quiet=FALSE){
  ZERO <- colSums(t(x==0),na.rm=TRUE)
  x <- x[which(ZERO<=Nmax),,drop=FALSE]
  if(!quiet){
    cat(report.dim(x,header=paste0("Pruning samples on zero values [Nmax>=",Nmax,"]")))
  }
  return(invisible(x))
}

impute.miss<-function(x){
  ## This is an boiler-plate solution :)
  N <- length(which(is.na(x)))
  PERC <- round(100*(N/(nrow(x)*ncol(x))),digits=3)
  cat("| Imputing ",N,"[",PERC,"%] missing values ... ")
  x[which(is.na(x))] <- 0
  return(x)
}