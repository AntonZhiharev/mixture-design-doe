# R-эталон для golden-тестов (REBUILD_SPEC §6), БОЕВОЙ вариант (q=5).
#
# Считается штатным R (base R: lm/qr/det/solve + явные формулы Деррингера-Сюича
# и GP-постериора Matern5/2 ARD) — БЕЗ внешних пакетов, чтобы не зависеть от
# сетевой установки. Обмен с Python — через CSV в каталоге io (аргумент 1).
#
# Эквивалент пакетов R (provenance в фикстурах):
#   Scheffe OLS         <- lm(y ~ terms - 1)          (mixexp::MixModel/lm)
#   D-/I-критерии        <- det(t(M)%*%M), solve+trace  (AlgDesign)
#   desirability         <- Derringer-Suich формулы     (desirability::dOverall)
#   GP Matern5/2 ARD     <- явный постериор при фикс. θ  (DiceKriging::km)
#
# Запуск: Rscript compute_reference.R <io_dir>

args <- commandArgs(trailingOnly = TRUE)
io <- if (length(args) >= 1) args[1] else "_io"

rd <- function(name) as.matrix(read.csv(file.path(io, name), header = FALSE))
rdv <- function(name) as.numeric(scan(file.path(io, name), quiet = TRUE, sep = ","))
wr <- function(name, x) write.table(x, file.path(io, name), sep = ",",
                                     row.names = FALSE, col.names = FALSE)

# --- модельная матрица Шеффе (linear + pairwise), порядок = Python combinations
build_scheffe <- function(X, order = 2) {
  q <- ncol(X); cols <- list()
  for (i in 1:q) cols[[length(cols) + 1]] <- X[, i]
  if (order >= 2) {
    pr <- combn(q, 2)
    for (k in 1:ncol(pr)) cols[[length(cols) + 1]] <- X[, pr[1, k]] * X[, pr[2, k]]
  }
  do.call(cbind, cols)
}

# ===================== Scheffe OLS (lm) =====================
X <- rd("scheffe_X.csv"); y <- rdv("scheffe_y.csv")
M <- build_scheffe(X, 2)
df <- data.frame(y = y, M)
fit <- lm(y ~ . - 1, data = df)               # OLS без свободного члена
beta <- as.numeric(coef(fit))
fitted <- as.numeric(fitted(fit))
resid <- y - fitted
n <- nrow(M); p <- ncol(M)
sse <- sum(resid^2); sst <- sum((y - mean(y))^2)
r2 <- 1 - sse / sst
adj_r2 <- 1 - (sse / (n - p)) / (sst / (n - 1))
rmse <- sqrt(sse / (n - p))
wr("out_scheffe_coef.csv", beta)
wr("out_scheffe_fitted.csv", fitted)
wr("out_scheffe_scalars.csv", c(r2, adj_r2, rmse))

# ===================== D-критерии (det) =====================
XtX <- t(M) %*% M
d_crit <- det(XtX)
d_eff <- (det(XtX))^(1 / p) / n
wr("out_dopt_scalars.csv", c(d_crit, d_eff))

# ===================== I-критерий (trace) =====================
W <- rd("iopt_W.csv")
i_crit <- sum(diag(solve(XtX, W)))
wr("out_iopt_scalar.csv", c(i_crit))

# ===================== desirability (Derringer-Suich) =====================
dp <- read.csv(file.path(io, "desir_params.csv"), header = TRUE)  # kind,low,high,target,s,s2,weight
ymax <- rdv("desir_y_max.csv"); ymin <- rdv("desir_y_min.csv"); ytgt <- rdv("desir_y_tgt.csv")
d_max <- with(dp[1, ], ifelse(ymax <= low, 0,
              ifelse(ymax >= high, 1, ((ymax - low) / (high - low))^s)))
d_min <- with(dp[2, ], ifelse(ymin <= low, 1,
              ifelse(ymin >= high, 0, ((high - ymin) / (high - low))^s)))
d_tgt <- with(dp[3, ], ifelse(ytgt < low | ytgt > high, 0,
              ifelse(ytgt <= target,
                     pmin(pmax((ytgt - low) / (target - low), 0), 1)^s,
                     pmin(pmax((high - ytgt) / (high - target), 0), 1)^s2)))
W3 <- dp$weight; W3 <- W3 / sum(W3)
D <- rbind(d_max, d_min, d_tgt)
overall <- numeric(ncol(D))
for (j in 1:ncol(D)) {
  col <- D[, j]
  overall[j] <- if (any(col <= 0)) 0 else exp(sum(W3 * log(pmin(pmax(col, 1e-300), 1))))
}
wr("out_desir_dmax.csv", d_max)
wr("out_desir_dmin.csv", d_min)
wr("out_desir_dtgt.csv", d_tgt)
wr("out_desir_overall.csv", overall)

# ===================== GP Matern5/2 ARD (фикс. θ) =====================
matern52 <- function(Xa, Xb, ls) {
  Xa <- sweep(Xa, 2, ls, "/"); Xb <- sweep(Xb, 2, ls, "/")
  a2 <- rowSums(Xa^2); b2 <- rowSums(Xb^2)
  d2 <- outer(a2, b2, "+") - 2 * Xa %*% t(Xb); d2[d2 < 0] <- 0
  r <- sqrt(d2)
  (1 + sqrt(5) * r + 5 / 3 * r^2) * exp(-sqrt(5) * r)
}
Xtr <- rd("gp_Xtrain.csv"); ytr <- rdv("gp_ytrain.csv"); Xte <- rd("gp_Xtest.csv")
gpar <- rdv("gp_params.csv")                 # const, noise
ls <- rdv("gp_ls.csv")
const <- gpar[1]; noise <- gpar[2]
K <- const * matern52(Xtr, Xtr, ls) + noise * diag(nrow(Xtr))
Ks <- const * matern52(Xtr, Xte, ls)
U <- chol(K)                                  # U^T U = K (U upper)
alpha <- backsolve(U, forwardsolve(t(U), ytr))
gp_mean <- as.numeric(t(Ks) %*% alpha)
v <- forwardsolve(t(U), Ks)
gp_var <- (const + noise) - colSums(v^2)
gp_var[gp_var < 0] <- 0
wr("out_gp_mean.csv", gp_mean)
wr("out_gp_sd.csv", sqrt(gp_var))

cat("R reference computed OK (q=", ncol(X), ", p=", p, ")\n", sep = "")
