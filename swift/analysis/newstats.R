library(truncnorm)

trpd <- function(a, b) list(truncnorm::dtruncnorm, a = a, b = b, mean = a+(b-a)/2, sd=(b-a)/2)

hpd.midpoint <- function(x, p = .95) {
  y <- as.mcmc(as.matrix(x))
  hpdi <- HPDinterval(y, prob = p)
  (hpdi[,"lower"] + hpdi[,"upper"]) / 2
}

read.corpus <- function(corpus_file) {
  corpus = read.table(corpus_file, sep="\t", row.names = NULL, header=F)
  colnames(corpus)[1:3] = c("nw","nl","wfreq")
  colnames(corpus)[4:ncol(corpus)] = paste0("V", 1:(ncol(corpus)-3))
  
  # where (at which line) does each sentence begin?
  corpus.itemstart <- integer(0)
  i <- 1
  while(i <= nrow(corpus)) {
    corpus.itemstart <- c(corpus.itemstart, i)
    i <- i + corpus$nw[i]
  }
  # note: length of corpus.itemstart equals number of sentences in corpus
  
  # repeat each sentence number (1..n) as often as there are words b/c each sentence has so many lines
  corpus$sno = rep(1L:length(corpus.itemstart), corpus$nw[corpus.itemstart])
  # only first word has $nw != NA, so set all words' $nw to the value that the first word in that sentence has
  corpus$nw = corpus$nw[corpus.itemstart[corpus$sno]]
  corpus$iw = do.call(.Primitive("c"), lapply(1L:length(corpus.itemstart), function(i) 1L:corpus$nw[corpus.itemstart[i]]))
  
  return(corpus)
}


fixseq_annotate <- function(fixseq, corpus) {
  trial_starts <- which(fixseq$first_last == 1)
  trial_ends <- which(fixseq$first_last == 2)
  stopifnot(length(trial_starts) == length(trial_ends))
  stopifnot(trial_starts < trial_ends)
  fixseq$trial_id <- NA_integer_
  fixseq$ifix <- NA_integer_
  fixseq$word_id <- match(paste(fixseq$sno, fixseq$fw), paste(corpus$sno, corpus$iw))
  for(i in seq_along(trial_starts)) {
    fixseq$trial_id[trial_starts[i]:trial_ends[i]] <- i
    fixseq$ifix[trial_starts[i]:trial_ends[i]] <- seq_len(trial_ends[i]-trial_starts[i]+1)
  }
  fixseq %>% group_by(trial_id, sno, subject) %>% mutate(new_location = fw != lag(fw, default = 0)) %>% mutate(event_id = ifix[which(new_location)[cumsum(new_location)]]) %>% mutate(
    event_len = vapply(1:n(), function(i) sum(event_id == event_id[i]), integer(1)),
    is_fwfix = fw == lag(fw) + 1,
    is_refix = fw == lag(fw),
    is_frefix = is_refix & fl > lag(fl),
    is_brefix = is_refix & !is_frefix,
    is_fwrefix = fw == lag(fw) & is_fwfix[event_id],
    is_ffwrefix = is_fwrefix & fl > lag(fl),
    is_bfwrefix = is_fwrefix & !is_frefix,
    is_reg = fw < lag(fw),
    is_skip = fw > lag(fw) + 1,
    is_singlefix = fw != lag(fw) & fw != lead(fw),
    path = c(1L, vapply(2:n(), function(i) {
      sum(is_reg[seq_len(i-1)], na.rm = TRUE) + 1L
    }, integer(1))),
    pass = c(1L, vapply(2:n(), function(i) {
      prev_ws <- 1:(i-1)
      target_ws <- 2:i
      if(is_refix[i])
        NA_integer_
      else
        sum(fw[prev_ws] < fw[target_ws] & fw[prev_ws] < fw[i] & fw[target_ws] >= fw[i] | fw[prev_ws] > fw[target_ws] & fw[prev_ws] > fw[i] & fw[target_ws] <= fw[i])
    }, integer(1)))
  ) %>% mutate(pass = ifelse(!is.na(is_reg) & is_reg & (is.na(lag(is_reg)) | !lag(is_reg)), lag(pass), pass)) %>% mutate(pass = pass[event_id]) %>% mutate(
    first_fix_dur = ifelse(new_location & !lead(is_refix, default = FALSE), tfix, NA),
    first_fwfix_dur = ifelse(is_fwfix & !lead(is_refix, default = FALSE), tfix, NA),
    single_fix_dur = ifelse(is_singlefix, tfix, NA),
    single_fwfix_dur = ifelse(is_fwfix & is_singlefix, tfix, NA),
    gaze_dur = vapply(1:n(), function(i) {
      if(new_location[i])
        sum(tfix[event_id == event_id[i]])
      else
        NA
    }, double(1)),
    refix_dur = ifelse(is_refix, tfix, NA),
    fwrefix_dur = ifelse(is_fwrefix, tfix, NA),
    reg_dur = ifelse(is_reg, tfix, NA)
  )
}


fixseq_stats_by_trial <- function(fixseq, corpus) {
  slen <- corpus$nw[c(1, which(corpus$sno != lag(corpus$sno)))]
  corpus <- do.call(rbind, lapply(unique(fixseq$trial_id), function(i) {
    cbind(subset(corpus, sno == first(fixseq$sno[fixseq$trial_id == i])), trial_id = i, subject = first(fixseq$subject[fixseq$trial_id == i]))
  }))
  fixseq_skippings <- do.call(rbind, lapply(fixseq %>% split(.$trial_id), function(trial) {
    skips <- which(trial$is_skip)
    do.call(rbind, lapply(skips, function(j) {
      skipped_words <- (trial$fw[j-1]+1):(trial$fw[j]-1)
      data.frame(
        trial_id = first(trial$trial_id), 
        subject = first(trial$subject),
        is_skipped = TRUE, 
        fw = skipped_words, 
        event_id = trial$event_id[j],
        pass = vapply(skipped_words, function(w) {
          wi <- which(trial$fw == w & trial$event_id < trial$event_id[j])
          if(length(wi) > 0)
            as.integer(max(trial$pass[wi])) + 1L
          else
            1L
        }, integer(1))
      )
    }))
  }))
  bind_rows(fixseq %>% mutate(is_skipped = FALSE) %>% ungroup() %>% select(-sno), fixseq_skippings) %>%
    mutate(is_fp = pass == 1) %>%
    right_join(corpus, by = c("trial_id" = "trial_id", "fw" = "iw", "subject" = "subject")) %>%
    subset(fw > 1 & fw < nw) %>%
    group_by(subject, sno, trial_id, fw, nl, wfreq) %>%
    summarize(
      first_fix_dur = ifelse(any(!is.na(first_fix_dur)), mean(first_fix_dur, na.rm = TRUE), NA),
      first_fwfix_dur = ifelse(any(!is.na(first_fwfix_dur)), mean(first_fwfix_dur, na.rm = TRUE), NA),
      total_viewing_time = ifelse(any(!is.na(gaze_dur)), sum(gaze_dur, na.rm = TRUE), NA),
      gaze_dur = ifelse(any(!is.na(gaze_dur)), mean(gaze_dur, na.rm = TRUE), NA),
      refix_dur = ifelse(any(!is.na(refix_dur)), mean(refix_dur, na.rm = TRUE), NA),
      fwrefix_dur = ifelse(any(!is.na(fwrefix_dur)), mean(fwrefix_dur, na.rm = TRUE), NA),
      reg_dur = ifelse(any(!is.na(reg_dur)), mean(reg_dur, na.rm = TRUE), NA),
      single_fix_dur = ifelse(any(!is.na(single_fix_dur)), mean(single_fix_dur, na.rm = TRUE), NA),
      single_fwfix_dur = ifelse(any(!is.na(single_fwfix_dur)), mean(single_fwfix_dur, na.rm = TRUE), NA),
      fp_first_fix_dur = ifelse(any(is_fp & !is.na(first_fix_dur)), mean(first_fix_dur[is_fp], na.rm = TRUE), NA),
      fp_first_fwfix_dur = ifelse(any(is_fp & !is.na(first_fwfix_dur)), mean(first_fwfix_dur[is_fp], na.rm = TRUE), NA),
      fp_gaze_dur = ifelse(any(is_fp & !is.na(gaze_dur)), mean(gaze_dur[is_fp], na.rm = TRUE), NA),
      fp_refix_dur = ifelse(any(is_fp & !is.na(refix_dur)), mean(refix_dur[is_fp], na.rm = TRUE), NA),
      fp_fwrefix_dur = ifelse(any(is_fp & !is.na(fwrefix_dur)), mean(fwrefix_dur[is_fp], na.rm = TRUE), NA),
      fp_reg_dur = ifelse(any(is_fp & !is.na(reg_dur)), mean(reg_dur[is_fp], na.rm = TRUE), NA),
      fp_single_fix_dur = ifelse(any(is_fp & !is.na(single_fix_dur)), mean(single_fix_dur[is_fp], na.rm = TRUE), NA),
      fp_single_fwfix_dur = ifelse(any(is_fp & !is.na(single_fwfix_dur)), mean(single_fwfix_dur[is_fp], na.rm = TRUE), NA),
      pass = max(pass),
      fix = sum(!is.na(tfix)),
      fwfix = sum(is_fwfix, na.rm = TRUE),
      refix = sum(is_refix, na.rm = TRUE),
      frefix = sum(is_frefix, na.rm = TRUE),
      brefix = sum(is_brefix, na.rm = TRUE),
      fwrefix = sum(is_fwrefix, na.rm = TRUE),
      ffwrefix = sum(is_ffwrefix, na.rm = TRUE),
      bfwrefix = sum(is_bfwrefix, na.rm = TRUE),
      regin = sum(is_reg, na.rm = TRUE),
      regout = sum(lead(is_reg), na.rm = TRUE),
      singlefix = sum(is_singlefix, na.rm = TRUE),
      skipped = sum(is_skipped, na.rm = TRUE),
      fp_fix = sum(!is.na(tfix) & is_fp),
      fp_fwfix = sum(is_fwfix & is_fp, na.rm = TRUE),
      fp_refix = sum(is_refix & is_fp, na.rm = TRUE),
      fp_frefix = sum(is_frefix & is_fp, na.rm = TRUE),
      fp_brefix = sum(is_brefix & is_fp, na.rm = TRUE),
      fp_fwrefix = sum(is_fwrefix & is_fp, na.rm = TRUE),
      fp_ffwrefix = sum(is_ffwrefix & is_fp, na.rm = TRUE),
      fp_bfwrefix = sum(is_bfwrefix & is_fp, na.rm = TRUE),
      fp_regin = sum(is_reg & is_fp, na.rm = TRUE),
      fp_regout = sum(lead(is_reg) & is_fp, na.rm = TRUE),
      fp_singlefix = sum(is_singlefix & is_fp, na.rm = TRUE),
      fp_skipped = sum(is_skipped & is_fp, na.rm = TRUE),
      regin_l2 = sum(is_reg & fw >= slen[sno] - 1, na.rm = TRUE),
      regout_l2 = sum(lead(is_reg) & fw >= slen[sno] - 1, na.rm = TRUE)
    ) %>%
    ungroup()
}

fixstats_stats_summary <- function(fixseq, durations = vars(first_fix_dur, first_fwfix_dur, gaze_dur, refix_dur, fwrefix_dur, reg_dur, fp_first_fix_dur, fp_first_fwfix_dur, fp_gaze_dur, fp_refix_dur, fp_fwrefix_dur, fp_reg_dur, fp_single_fix_dur, fp_single_fwfix_dur, total_viewing_time, single_fix_dur, single_fwfix_dur), probs = vars(regin, regout, refix, fwrefix, singlefix, skipped, fp_regin, fp_regout, fp_refix, fp_fwrefix, fp_singlefix, fp_skipped, regin_l2, regout_l2)) {
  fixdurs <- fixseq %>% 
    summarize_at(
      durations, 
      list(m = ~ if(any(!is.na(.))) mean(., na.rm = TRUE) else NA)
    )
  fixprobs <- fixseq %>%
    summarize_at(
      probs,
      list(p = ~ if(any(!is.na(.))) mean(. > .5, na.rm = TRUE) else 0, n = ~ if(any(!is.na(.))) sum(., na.rm = TRUE) else 0)
    )
  fixdurs %>% inner_join(fixprobs, by = sapply(groups(fixseq), as.character))
}


