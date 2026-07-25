#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readxl)
  library(ggplot2)
  library(jsonlite)
})

RAW_XLS <- "data/raw/default of credit card clients.xls"
RAW_ZIP <- "data/raw/default-of-credit-card-clients.zip"
CANONICAL_CSV <- "data/processed/credit_default_validated.csv"
SPLIT_CSV <- "data/processed/split_assignments.csv"
SUMMARY_JSON <- "reports/day2_data_quality_summary.json"
SEED <- 42L
EXPECTED_XLS_SHA256 <- "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
EXPECTED_ZIP_SHA256 <- "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
SPLIT_PROPORTIONS <- c(train = 0.60, validation = 0.20, test = 0.20)
SPLIT_TOLERANCE <- 0.01
PREVALENCE_TOLERANCE <- 0.005
ORIGINAL_PREDICTORS <- c(
  "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
  "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
  paste0("BILL_AMT", 1:6), paste0("PAY_AMT", 1:6)
)

sha256 <- function(path) {
  output <- system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE)
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}

stopifnot(file.exists(RAW_XLS), file.exists(RAW_ZIP))
actual_xls_sha256 <- sha256(RAW_XLS)
actual_zip_sha256 <- sha256(RAW_ZIP)
stopifnot(identical(actual_xls_sha256, EXPECTED_XLS_SHA256))
stopifnot(identical(actual_zip_sha256, EXPECTED_ZIP_SHA256))

load_raw_credit_data <- function(path = RAW_XLS) {
  raw <- read_excel(path, sheet = "Data", col_names = FALSE, .name_repair = "minimal")
  stopifnot(nrow(raw) == 30002L, ncol(raw) == 25L)

  raw_codes <- as.character(unlist(raw[1, ], use.names = FALSE))
  descriptive_names <- as.character(unlist(raw[2, ], use.names = FALSE))
  raw_codes[[1]] <- "ID"
  raw_codes[[25]] <- "Y"
  stopifnot(identical(raw_codes, c("ID", paste0("X", 1:23), "Y")))

  data <- raw[-c(1, 2), , drop = FALSE]
  names(data) <- c(
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    paste0("BILL_AMT", 1:6), paste0("PAY_AMT", 1:6),
    "DEFAULT_NEXT_MONTH"
  )
  data[] <- lapply(data, function(x) as.numeric(as.character(x)))
  attr(data, "raw_codes") <- raw_codes
  attr(data, "descriptive_names") <- descriptive_names
  data
}

map_known_categories <- function(data) {
  sex_labels <- c(`1` = "Male", `2` = "Female")
  education_labels <- c(
    `1` = "Graduate school", `2` = "University",
    `3` = "High school", `4` = "Other"
  )
  marriage_labels <- c(`1` = "Married", `2` = "Single", `3` = "Other")
  repayment_labels <- c(
    `-1` = "Paid duly", `1` = "Delay: 1 month", `2` = "Delay: 2 months",
    `3` = "Delay: 3 months", `4` = "Delay: 4 months",
    `5` = "Delay: 5 months", `6` = "Delay: 6 months",
    `7` = "Delay: 7 months", `8` = "Delay: 8 months",
    `9` = "Delay: 9+ months"
  )

  map_or_unknown <- function(x, labels) {
    mapped <- unname(labels[as.character(x)])
    mapped[is.na(mapped)] <- "Unknown/Other"
    mapped
  }

  data$SEX_CATEGORY <- map_or_unknown(data$SEX, sex_labels)
  data$EDUCATION_CATEGORY <- map_or_unknown(data$EDUCATION, education_labels)
  data$MARRIAGE_CATEGORY <- map_or_unknown(data$MARRIAGE, marriage_labels)
  for (field in c("PAY_0", paste0("PAY_", 2:6))) {
    data[[paste0(field, "_CATEGORY")]] <- map_or_unknown(data[[field]], repayment_labels)
  }
  data
}

predictor_profile_keys <- function(data) {
  stopifnot(identical(ORIGINAL_PREDICTORS, names(data)[names(data) %in% ORIGINAL_PREDICTORS]))
  do.call(paste, c(data[ORIGINAL_PREDICTORS], sep = "\037"))
}

make_grouped_stratified_splits <- function(data, seed = SEED) {
  set.seed(seed)
  keys <- predictor_profile_keys(data)
  group_index <- match(keys, unique(keys))
  group_rows <- as.integer(table(group_index))
  group_defaults <- as.integer(tapply(data$DEFAULT_NEXT_MONTH, group_index, sum))
  groups <- data.frame(
    group_index = seq_along(group_rows),
    rows = group_rows,
    defaults = group_defaults,
    random_order = sample.int(length(group_rows)),
    stringsAsFactors = FALSE
  )
  groups <- groups[order(-groups$rows, -abs(groups$defaults / groups$rows - mean(data$DEFAULT_NEXT_MONTH)), groups$random_order), ]

  target_rows <- SPLIT_PROPORTIONS * nrow(data)
  target_defaults <- SPLIT_PROPORTIONS * sum(data$DEFAULT_NEXT_MONTH)
  target_non_defaults <- SPLIT_PROPORTIONS * sum(data$DEFAULT_NEXT_MONTH == 0)
  current_rows <- setNames(rep(0, 3), names(SPLIT_PROPORTIONS))
  current_defaults <- setNames(rep(0, 3), names(SPLIT_PROPORTIONS))
  group_partition <- rep(NA_character_, nrow(groups))

  objective <- function(rows, defaults) {
    non_defaults <- rows - defaults
    sum(((defaults - target_defaults) / target_defaults)^2) +
      sum(((non_defaults - target_non_defaults) / target_non_defaults)^2)
  }

  for (i in seq_len(nrow(groups))) {
    candidate_order <- sample(names(SPLIT_PROPORTIONS))
    scores <- vapply(candidate_order, function(candidate) {
      proposed_rows <- current_rows
      proposed_defaults <- current_defaults
      proposed_rows[[candidate]] <- proposed_rows[[candidate]] + groups$rows[[i]]
      proposed_defaults[[candidate]] <- proposed_defaults[[candidate]] + groups$defaults[[i]]
      objective(proposed_rows, proposed_defaults)
    }, numeric(1))
    selected <- candidate_order[[which.min(scores)]]
    group_partition[[groups$group_index[[i]]]] <- selected
    current_rows[[selected]] <- current_rows[[selected]] + groups$rows[[i]]
    current_defaults[[selected]] <- current_defaults[[selected]] + groups$defaults[[i]]
  }

  split_label <- group_partition[group_index]
  stopifnot(!anyNA(split_label))
  data.frame(ID = data$ID, split = split_label, stringsAsFactors = FALSE)
}

frequency_records <- function(data, field, documented_values) {
  counts <- as.data.frame(table(data[[field]], useNA = "ifany"), stringsAsFactors = FALSE)
  names(counts) <- c("value", "count")
  counts$field <- field
  counts$documented <- counts$value %in% as.character(documented_values)
  counts[, c("field", "value", "count", "documented")]
}

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)

raw_data <- load_raw_credit_data()
validated_data <- map_known_categories(raw_data)
splits <- make_grouped_stratified_splits(raw_data)

stopifnot(nrow(raw_data) == 30000L, ncol(raw_data) == 25L)
stopifnot(identical(sort(raw_data$ID), as.numeric(1:30000)))
stopifnot(!anyDuplicated(raw_data$ID))
stopifnot(all(raw_data$DEFAULT_NEXT_MONTH %in% c(0, 1)))

split_counts <- as.data.frame(table(splits$split), stringsAsFactors = FALSE)
names(split_counts) <- c("split", "count")
stopifnot(identical(sort(as.character(split_counts$split)), sort(names(SPLIT_PROPORTIONS))))
actual_proportions <- setNames(split_counts$count, split_counts$split) / nrow(raw_data)
stopifnot(all(abs(actual_proportions[names(SPLIT_PROPORTIONS)] - SPLIT_PROPORTIONS) <= SPLIT_TOLERANCE))

write.csv(validated_data, CANONICAL_CSV, row.names = FALSE, na = "")
write.csv(splits[order(splits$ID), ], SPLIT_CSV, row.names = FALSE)

category_frequencies <- do.call(rbind, list(
  frequency_records(raw_data, "SEX", c(1, 2)),
  frequency_records(raw_data, "EDUCATION", c(1, 2, 3, 4)),
  frequency_records(raw_data, "MARRIAGE", c(1, 2, 3)),
  frequency_records(raw_data, "PAY_0", c(-1, 1:9)),
  frequency_records(raw_data, "PAY_2", c(-1, 1:9)),
  frequency_records(raw_data, "PAY_3", c(-1, 1:9)),
  frequency_records(raw_data, "PAY_4", c(-1, 1:9)),
  frequency_records(raw_data, "PAY_5", c(-1, 1:9)),
  frequency_records(raw_data, "PAY_6", c(-1, 1:9))
))

numeric_fields <- c("LIMIT_BAL", "AGE", paste0("BILL_AMT", 1:6), paste0("PAY_AMT", 1:6))
numeric_ranges <- data.frame(
  field = numeric_fields,
  minimum = vapply(raw_data[numeric_fields], min, numeric(1), na.rm = TRUE),
  q1 = vapply(raw_data[numeric_fields], quantile, numeric(1), probs = 0.25, na.rm = TRUE),
  median = vapply(raw_data[numeric_fields], median, numeric(1), na.rm = TRUE),
  q3 = vapply(raw_data[numeric_fields], quantile, numeric(1), probs = 0.75, na.rm = TRUE),
  maximum = vapply(raw_data[numeric_fields], max, numeric(1), na.rm = TRUE),
  row.names = NULL
)

all_duplicate_rows <- sum(duplicated(raw_data))
feature_duplicate_rows <- sum(duplicated(raw_data[setdiff(names(raw_data), "ID")]))
missing_by_field <- colSums(is.na(raw_data))
target_counts <- as.data.frame(table(raw_data$DEFAULT_NEXT_MONTH), stringsAsFactors = FALSE)
names(target_counts) <- c("target", "count")
target_counts$percentage <- round(100 * target_counts$count / sum(target_counts$count), 4)

split_audit <- merge(splits, raw_data[c("ID", "DEFAULT_NEXT_MONTH")], by = "ID")
split_balance <- as.data.frame(table(split_audit$split, split_audit$DEFAULT_NEXT_MONTH), stringsAsFactors = FALSE)
names(split_balance) <- c("split", "target", "count")
split_totals <- aggregate(count ~ split, split_balance, sum)
names(split_totals)[[2]] <- "split_total"
split_balance <- merge(split_balance, split_totals, by = "split")
split_balance$percentage <- round(100 * split_balance$count / split_balance$split_total, 4)

profile_keys <- predictor_profile_keys(raw_data)
duplicate_profile_groups <- split(raw_data$ID[duplicated(profile_keys) | duplicated(profile_keys, fromLast = TRUE)],
                                  profile_keys[duplicated(profile_keys) | duplicated(profile_keys, fromLast = TRUE)])
split_lookup <- setNames(splits$split, splits$ID)
cross_split_duplicate_groups <- Filter(
  function(ids) length(unique(split_lookup[as.character(ids)])) > 1,
  duplicate_profile_groups
)
stopifnot(length(cross_split_duplicate_groups) == 0L)

full_prevalence <- mean(raw_data$DEFAULT_NEXT_MONTH)
partition_prevalence <- tapply(split_audit$DEFAULT_NEXT_MONTH, split_audit$split, mean)
stopifnot(all(abs(partition_prevalence - full_prevalence) <= PREVALENCE_TOLERANCE))

anomaly_counts <- list(
  non_positive_credit_limits = sum(raw_data$LIMIT_BAL <= 0),
  age_outside_18_to_100 = sum(raw_data$AGE < 18 | raw_data$AGE > 100),
  negative_bill_amounts_by_field = as.list(vapply(raw_data[paste0("BILL_AMT", 1:6)], function(x) sum(x < 0), integer(1))),
  zero_bill_amounts_by_field = as.list(vapply(raw_data[paste0("BILL_AMT", 1:6)], function(x) sum(x == 0), integer(1))),
  negative_payment_amounts_by_field = as.list(vapply(raw_data[paste0("PAY_AMT", 1:6)], function(x) sum(x < 0), integer(1))),
  zero_payment_amounts_by_field = as.list(vapply(raw_data[paste0("PAY_AMT", 1:6)], function(x) sum(x == 0), integer(1)))
)

# Exploratory figures use training rows only. Test labels are consulted solely for
# the mechanical stratification audit above, never for feature or design choices.
train_ids <- splits$ID[splits$split == "train"]
train_data <- raw_data[raw_data$ID %in% train_ids, , drop = FALSE]

plot_target <- as.data.frame(table(train_data$DEFAULT_NEXT_MONTH), stringsAsFactors = FALSE)
names(plot_target) <- c("Default", "Customers")
plot_target$Default <- factor(plot_target$Default, levels = c("0", "1"), labels = c("No default", "Default"))
p1 <- ggplot(plot_target, aes(Default, Customers, fill = Default)) +
  geom_col(width = 0.62, show.legend = FALSE) +
  geom_text(aes(label = paste0(Customers, " (", round(100 * Customers / sum(Customers), 1), "%)")), vjust = -0.4) +
  scale_fill_manual(values = c("#4C78A8", "#E45756")) +
  labs(title = "Training-set target balance", subtitle = "Final test labels excluded from exploratory analysis", x = NULL, y = "Customers") +
  theme_minimal(base_size = 11) + expand_limits(y = max(plot_target$Customers) * 1.1)
ggsave("reports/figures/day2_training_target_balance.png", p1, width = 7, height = 4.5, dpi = 160)

pay_fields <- c("PAY_0", paste0("PAY_", 2:6))
pay_long <- do.call(rbind, lapply(pay_fields, function(field) {
  out <- as.data.frame(table(train_data[[field]]), stringsAsFactors = FALSE)
  names(out) <- c("Code", "Customers")
  out$Month <- field
  out
}))
p2 <- ggplot(pay_long, aes(Code, Customers, fill = Month)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Month, ncol = 3, scales = "free_y") +
  labs(title = "Training-set repayment-status codes", subtitle = "Codes -2 and 0 are undocumented by the UCI catalogue", x = "Raw code", y = "Customers") +
  theme_minimal(base_size = 10)
ggsave("reports/figures/day2_repayment_code_frequencies.png", p2, width = 9, height = 6, dpi = 160)

audit_fields <- c("SEX", "EDUCATION", "MARRIAGE")
audit_long <- do.call(rbind, lapply(audit_fields, function(field) {
  out <- as.data.frame(table(train_data[[field]]), stringsAsFactors = FALSE)
  names(out) <- c("Code", "Customers")
  out$Field <- field
  out
}))
p3 <- ggplot(audit_long, aes(Code, Customers, fill = Field)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Field, scales = "free") +
  labs(title = "Training-set audit-variable codes", subtitle = "Audit-only variables are excluded from primary model features", x = "Raw code", y = "Customers") +
  theme_minimal(base_size = 11)
ggsave("reports/figures/day2_audit_category_frequencies.png", p3, width = 8, height = 4.5, dpi = 160)

summary <- list(
  generated_at = format(Sys.time(), tz = "Europe/London", usetz = TRUE),
  random_seed = SEED,
  split_method = list(
    description = "Deterministic group-aware approximate stratification using all 23 original predictors; ID and target excluded from profile key.",
    target_proportions = as.list(SPLIT_PROPORTIONS),
    row_proportion_absolute_tolerance = SPLIT_TOLERANCE,
    prevalence_absolute_tolerance = PREVALENCE_TOLERANCE
  ),
  raw_files = list(
    xls = list(path = RAW_XLS, sha256 = actual_xls_sha256),
    zip = list(path = RAW_ZIP, sha256 = actual_zip_sha256)
  ),
  raw_workbook_structure = list(rows_including_two_header_rows = 30002, columns = 25, sheet = "Data"),
  dataset = list(rows = nrow(raw_data), columns = ncol(raw_data), column_names = names(raw_data)),
  data_types = as.list(setNames(rep("numeric", ncol(raw_data)), names(raw_data))),
  id_unique = !anyDuplicated(raw_data$ID),
  exact_duplicate_rows_including_id = all_duplicate_rows,
  exact_duplicate_rows_excluding_id = feature_duplicate_rows,
  predictor_profile_groups = length(unique(profile_keys)),
  duplicate_profile_groups = length(duplicate_profile_groups),
  rows_in_duplicate_profile_groups = sum(lengths(duplicate_profile_groups)),
  cross_split_duplicate_profile_groups = length(cross_split_duplicate_groups),
  rows_in_cross_split_duplicate_profile_groups = sum(lengths(cross_split_duplicate_groups)),
  missing_by_field = as.list(missing_by_field),
  target_distribution_full_dataset_for_reporting = target_counts,
  category_frequencies = category_frequencies,
  undocumented_category_frequencies = category_frequencies[!category_frequencies$documented, ],
  numeric_ranges = numeric_ranges,
  anomaly_counts = anomaly_counts,
  split_balance_integrity_audit = split_balance,
  final_test_label_policy = "Test labels used only to create and mechanically verify stratification; not used for exploratory feature or design decisions.",
  primary_feature_exclusions = c("ID", "SEX", "AGE", "MARRIAGE", "EDUCATION"),
  figures = c(
    "reports/figures/day2_training_target_balance.png",
    "reports/figures/day2_repayment_code_frequencies.png",
    "reports/figures/day2_audit_category_frequencies.png"
  )
)
write_json(summary, SUMMARY_JSON, pretty = TRUE, auto_unbox = TRUE, na = "null")

cat("Day 2 preparation complete\n")
cat("Rows:", nrow(raw_data), "Columns:", ncol(raw_data), "\n")
cat("ID unique:", !anyDuplicated(raw_data$ID), "\n")
cat("Exact duplicates including ID:", all_duplicate_rows, "\n")
cat("Exact duplicates excluding ID:", feature_duplicate_rows, "\n")
cat("Predictor-profile groups:", length(unique(profile_keys)), "\n")
cat("Duplicate predictor-profile groups:", length(duplicate_profile_groups), "\n")
cat("Predictor-profile groups crossing splits:", length(cross_split_duplicate_groups), "groups /", sum(lengths(cross_split_duplicate_groups)), "rows\n")
cat("Missing values:", sum(missing_by_field), "\n")
print(target_counts)
print(category_frequencies[!category_frequencies$documented, ])
print(numeric_ranges)
print(split_balance[order(split_balance$split, split_balance$target), ])
