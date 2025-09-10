#!/opt/software/R/bin/Rscript --vanilla

rm(list = ls())
.libPaths("~/ukb/Scripts_Lieke/Library")

library("dplyr")
library("data.table")
library("tidyselect")
library("mice")
library("stringr")

UKB <- fread("~/ukb/data/67864-2009984-45839/csv/ukb45839.csv")

#MetaboHealth <- read.csv("Roundboth_MiMIR_MetaboHealth.csv",row.names=1)
#UKB = subset(UKB, eid %in% MetaboHealth$eid)

UKB_Lifestyle <- select(UKB, c(eid, starts_with("48-"), starts_with("21001-"), starts_with("22038-"), starts_with("22039-"), starts_with("22033-"), starts_with("22034-"), starts_with("1239-"), starts_with("1558-"), starts_with("1160-"), starts_with("1289-"), starts_with("1299-"), starts_with("1309-"), starts_with("1319-"), starts_with("1329-"), starts_with("1339-"), starts_with("1349-"), starts_with("1369-"),starts_with("1379-"), starts_with("1389-"), starts_with("1568-"), starts_with("1578-"), starts_with("1588-"), starts_with("1598-"), starts_with("1608"), starts_with("5364"), starts_with("3456"),starts_with("23655"), starts_with("23649"), starts_with("23650"), starts_with("23651"), starts_with("23652"), starts_with("23653"), starts_with("23654"), starts_with("30180"), starts_with("30190"), starts_with("22035-"), starts_with("1438-"), starts_with("1448-"),starts_with("1458-"), starts_with("1468-"), starts_with("4407-"), starts_with("4418-"), starts_with("4429-"), starts_with("4440-"), starts_with("4451-"), starts_with("4462-")))

ses <- select(UKB, c(eid, starts_with("6138-")))
ses <- ses %>%
  mutate(
    ses.0.0 = case_when(
      if_any(starts_with("6138-0."), ~ .x == -7) ~ 0,
      if_any(starts_with("6138-0."), ~ .x == 1) ~ 3,
      if_any(starts_with("6138-0."), ~ .x %in% c(2, 6)) ~ 2,
      if_any(starts_with("6138-0."), ~ .x %in% c(3, 4, 5)) ~ 1,
      TRUE ~ NA_real_
    ),
    ses.1.0 = case_when(
      if_any(starts_with("6138-1."), ~ .x == -7) ~ 0,
      if_any(starts_with("6138-1."), ~ .x == 1) ~ 3,
      if_any(starts_with("6138-1."), ~ .x %in% c(2, 6)) ~ 2,
      if_any(starts_with("6138-1."), ~ .x %in% c(3, 4, 5)) ~ 1,
      TRUE ~ NA_real_
    ),
  )

rm(UKB)
ses = select(ses, c(eid, ses.0.0, ses.1.0))

ukb <- fread("/data/ukb/data/67864-2009984-45839/csv/ukb45839.csv")
#ukb <- select(ukb, c(eid, starts_with("3166")))
ukb <- select(ukb, c(eid, starts_with("52-"),starts_with("53-"), starts_with("55-"), starts_with("34-"), starts_with("31-")))
names(ukb)[2:ncol(ukb)] = paste0("x", gsub("-", ".", names(ukb)[2:ncol(ukb)]))
ukb <- ukb %>% mutate(
  month = case_when(x52.0.0 < 10 ~ paste0(0,x52.0.0),
                    TRUE ~ as.character(x52.0.0)),
  year = x34.0.0,
  dob = lubridate::ymd(paste(year,month,"15", sep = "-")),
  sex =  case_when(x31.0.0 == 0 ~ "w",
                   x31.0.0 == 1 ~ "m"))
ukb <- ukb %>%
  setNames(gsub("^x55.", "ssn.", names(.))) %>%#Month of visiting assessment center
  setNames(gsub("^x53.", "centerdate.", names(.)))#Date of visiting assessment cente

ukb = select(ukb, c(eid, dob, sex, starts_with(c("ssn.", "centerdate."))))
fwrite(ukb, "basic.csv")

UKB_Lifestyle = full_join(UKB_Lifestyle, ukb)

rm(ukb)

UKB_Lifestyle <- UKB_Lifestyle %>%
 setNames(gsub("^48-",   "waist.", names(.))) %>%
 setNames(gsub("^21001-","bmi.", names(.))) %>%
 setNames(gsub("^22035-","actguidl.", names(.))) %>%
 setNames(gsub("^22038-","metmod.", names(.))) %>%
 setNames(gsub("^22039-","metvig.", names(.))) %>%
 setNames(gsub("^22033-","actday.", names(.))) %>%
 setNames(gsub("^22034-","actmin.", names(.))) %>%
 setNames(gsub("^20116-","smoking3.", names(.))) %>%
 setNames(gsub("^1239-", "smoking.", names(.))) %>%
 setNames(gsub("^3456-", "smkfrq.", names(.))) %>%
 setNames(gsub("^1558-", "alcohol.", names(.))) %>%
 setNames(gsub("^1160-", "sleepdr.", names(.))) %>%
 setNames(gsub("^1289-", "veggies.", names(.))) %>%
 setNames(gsub("^1299-", "salad.", names(.))) %>%
 setNames(gsub("^1309-", "frfruit.", names(.))) %>%
 setNames(gsub("^1319-", "drfruit.", names(.))) %>%
 setNames(gsub("^1329-", "oilfish.", names(.))) %>%
 setNames(gsub("^1339-", "noilfish.", names(.))) %>%
 setNames(gsub("^1349-", "prmeat.", names(.))) %>%
 setNames(gsub("^1369-", "beef.", names(.))) %>%
 setNames(gsub("^1379-", "lamb.", names(.))) %>%
 setNames(gsub("^1389-", "pork.", names(.))) %>% 
 setNames(gsub("^1438-", "breadslices.", names(.)))%>% 
 setNames(gsub("^1448-", "breadtype.", names(.)))%>% 
 setNames(gsub("^1458-", "cerealbowl.", names(.)))%>% 
 setNames(gsub("^1468-", "cerealtype.", names(.)))%>% 
 setNames(gsub("^1568-", "rwine_week.", names(.))) %>% #Red wine (weekly average)
 setNames(gsub("^4407-", "rwine_month.", names(.))) %>% #Red wine (monthly average),
 setNames(gsub("^1578-", "champ_week.", names(.))) %>% #Champagne and white wine (weekly average)
 setNames(gsub("^4418-", "champ_month.", names(.))) %>% #Champagne and white wine (monthly average)
 setNames(gsub("^1588-", "beer_week.", names(.))) %>% #Beer and cider (weekly average)
 setNames(gsub("^4429-", "beer_month.", names(.))) %>% #Beer and cider (monthly average)
 setNames(gsub("^1598-", "spirit_week.", names(.))) %>% #Spirits (weekly average)
 setNames(gsub("^4440-", "spirit_month.", names(.))) %>% #Spirits (monthly average)
 setNames(gsub("^1608-", "fwine_week.", names(.))) %>% #Fortified wine (weekly average)
 setNames(gsub("^4451-", "fwine_month.", names(.))) %>% #Fortified wine (monthly average)
 setNames(gsub("^5364-", "othalc_week.", names(.))) %>% #Other alcoholic drinks (weekly average)
 setNames(gsub("^4462-", "othalc_month.", names(.))) %>% #Other alcoholic drinks (monthly average)
 setNames(gsub("^30180-","lymf.", names(.))) %>% # Lymfocytes
 setNames(gsub("^30190-","mono.", names(.)))  #Monocytes 
 
#basic=read.csv("basic.csv")

#UKB_Lifestyle = inner_join(UKB_Lifestyle, basic)

#set negative values veggies and fruit -less than once a day- to 0.5 in all other fields negative values represents missings (see next step)
UKB_Lifestyle <- UKB_Lifestyle %>% 
  mutate(across(
    starts_with(c("veggies.", "salad.", "frfruit.", "drfruit.", "smkfrq.")),
    ~ case_when(
        . == -10 ~ 0.5,
        . %in% c(-1, -3) ~ NA_real_,
        TRUE ~ .  # Keep other values unchanged
      )
  ))

# set values below zero to NA
UKB_Lifestyle <- UKB_Lifestyle %>% mutate_if(is.numeric, ~replace(., . < 0, NA))

#Citation: Liu, W.; Wang, T.; Zhu, M.; Jin, G. Healthy Diet, Polygenic Risk Score, and Upper Gastrointestinal Cancer Risk: A Prospective Study from UK Biobank. Nutrients 2023, 15, 1344. https://doi.org/10.3390/nu15061344
  
for(i in c(".0.0", ".1.0", ".2.0", ".3.0")){
  df = select(UKB_Lifestyle, c(eid, dob, ends_with(i)))
  df = df %>% rename_with(~str_remove(., i))
  df = df %>% mutate(
    alcperweek = case_when(alcohol == 1 | alcohol == 2 | alcohol == 3~ 1, #drinks alcohol daily, 2-4 times a week, 1-2 times a week
                           is.na(alcohol) ~ NA_real_,
                           TRUE ~ 0),
    rwine = case_when(!is.na(rwine_week) ~ 1.5 * rwine_week/7,
                      !is.na(rwine_month)~ 1.5 * rwine_month * 12 /365.25,
                      TRUE ~ 0),
    champ = case_when(!is.na(champ_week) ~ 1.5 * champ_week/7,
                      !is.na(champ_month)~ 1.5 * champ_month * 12 /365.25,
                      TRUE ~ 0),
    beer =  case_when(!is.na(beer_week) ~ 2.5 * beer_week/7,
                      !is.na(beer_month)~ 2.5 * beer_month * 12 /365.25,
                      TRUE ~ 0),
    spirit =case_when(!is.na(spirit_week) ~ 1.5 * rwine_week/7,
                      !is.na(spirit_month)~ 1.5 * rwine_month * 12 /365.25,
                      TRUE ~ 0),
    fwine = case_when(!is.na(fwine_week) ~ 1 * fwine_week/7,
                      !is.na(fwine_month)~ 1 * fwine_month * 12 /365.25,
                      TRUE ~ 0),
    othalc = case_when(!is.na(othalc_week) ~ 1 * othalc_week/7,
                       !is.na(othalc_month)~ 1 * othalc_month * 12 /365.25,
                       TRUE ~ 0),
    alcfreq = case_when(alcperweek == 1  ~ (rwine + champ + beer+ spirit +fwine+othalc),
                        alcperweek == 0 ~ 0,
                        TRUE ~ NA_real_),
    fruitscr = case_when(frfruit + drfruit/2  >= 4.0 ~ 1,
               frfruit + drfruit/2   < 4.0 ~ 0),
    vegscr = case_when(veggies/2 + salad/2 >= 4.0 ~ 1,
                       veggies/2 + salad/2< 4.0 ~ 0),
    fishscr = case_when(oilfish + noilfish >= 2 ~ 1,
                        oilfish + noilfish < 2  ~ 0),
    procmeatscr = case_when(prmeat <= 2 ~ 1,
                            prmeat > 2 ~ 0),
    rmeatscr= case_when(beef >= 3 | lamb >= 3 | pork >= 3 ~ 0,
                        beef == 2 &  (lamb ==  2 | pork == 2) ~ 0,
                        (beef == 2 | lamb ==  2) & pork == 2 ~ 0,
                        lamb == 2 &  (beef ==  2 | pork == 2) ~ 0,
                        !is.na(prmeat) & !is.na(beef) & !is.na(lamb) & !is.na(pork) ~1 ),
  wholegrainscr = case_when(breadtype == 3& breadslices/7 >= 3 ~ 1,
                         cerealtype !=2 & cerealtype != 5 & cerealbowl/7 >= 3 ~ 1,
                         breadtype == 3 & cerealtype !=2 & cerealtype != 5 & (breadslices + cerealbowl)/7 > 3 ~ 1,
                         !is.na(breadslices) & !is.na(cerealbowl) ~ 0),
  refinedscr = case_when(breadtype != 3 & breadslices/7 > 1.5 ~ 0,
                           (cerealtype ==2 | cerealtype == 5) & cerealbowl/7 > 1.5 ~ 0,
                           breadtype != 3 & (cerealtype ==2 | cerealtype == 5) & (breadslices + cerealbowl)/7 > 1.5 ~ 0,
                           !is.na(breadslices) & !is.na(cerealbowl) ~ 1),
    dietscore = fruitscr + vegscr+ fishscr + procmeatscr + rmeatscr +wholegrainscr +  refinedscr,
  season= case_when( ssn == "12"| ssn == "1" | ssn == "2" ~ "winter",
                     ssn == "3" | ssn == "4" | ssn == "5" ~ "spring",
                     ssn == "6" | ssn == "7" | ssn == "8" ~ "summer",
                     ssn == "9" | ssn == "10"| ssn == "11"~ "autumn"),
  season = as.factor(season),
  age= as.numeric(as.Date(centerdate) - as.Date(dob))/365.25,
  smokingfrq = case_when(smoking== 0 ~ 0,
                         smoking== 2 ~ 1,
                         smkfrq > 0 & smkfrq < 10 ~ 2,
                         smkfrq>= 10& smkfrq < 20 ~ 3,
                         smkfrq > 20 ~ 4),
  smokingyn = case_when(smoking == 1 | smoking ==2 ~ 1,
                        smoking ==0 ~ 0),
  sleepH = case_when(sleepdr > 8 ~ 1,
                     sleepdr >=7 & sleepdr <= 8 ~ 0),
  sleepS = case_when(sleepdr < 7 ~ 1,
                     sleepdr >=7 & sleepdr <= 8 ~ 0),
  sleep = case_when(sleepdr >=7 & sleepdr <= 8 ~ 0,
                    sleepdr < 7 ~ 1,
                    sleepdr > 8 ~ 2) )
  
  if(i == ".0.0"){
    df <- df %>% mutate(
      metmin = metvig + metmod
    )
  }
  
  df <- df %>% 
    setNames(ifelse(names(.) == "eid", names(.), paste0(names(.), i)))
  df = subset(df, rowSums(is.na(df)) < nrow(df) -1)
  
  UKB_Lifestyle = left_join(UKB_Lifestyle, df)
  
  
}
UKB_Lifestyle = left_join(UKB_Lifestyle, ses)

data.table::fwrite(UKB_Lifestyle, "UKB_Lifestyle.csv")

