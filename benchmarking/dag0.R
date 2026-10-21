## DAG0 comparator for the simulation study (paper Section "Benchmarking against
## other methods"). Runs the one-sample DAG0 sampler of the BayesDAG0 package
## (https://github.com/junsoukchoi/BayesDAG0) on one simulated dataset and writes
## the posterior inclusion probability (PIP) of every ordered pair.
##
## Usage:  Rscript dag0.R <X.csv> <out_prefix>
## Writes: <out_prefix>_post.csv   (n, n) PIPs, entry (i, j) = P(edge i -> j)

library(BayesDAG0)

set.seed(1)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L)
  stop("usage: Rscript dag0.R <X.csv> <out_prefix>")
X_file   <- args[1]
out_pref <- args[2]

## ---------------------------------------------------------------------------
## Sampler settings matched to the SPARK simulation runs (supplement table
## "Sampler settings for the simulation study"): 10 chains, 250,000 iterations of
## which the first 150,000 are burn-in, swap probability 0.2, and temperatures
## T_l = 2.5^((l - 1) / 9), i.e. inverse temperatures beta in [0.40, 1].
## ---------------------------------------------------------------------------
N_SAMPLE  <- 250000   # total iterations
N_BURNIN  <- 150000   # discarded, leaving 100,000 retained draws
N_CHAIN   <- 10
PROB_SWAP <- 0.2
TEMPERATURE <- 2.5^((seq_len(N_CHAIN) - 1) / (N_CHAIN - 1))

dat <- as.matrix(read.csv(X_file, row.names = 1))
p   <- ncol(dat)
cat(sprintf("read %s: %d cells x %d nodes\n", X_file, nrow(dat), p))

## ---------------------------------------------------------------------------
## Starting values: the empty graph, as for SPARK, with fixed values for the other
## parameters.
## ---------------------------------------------------------------------------
starting <- list()
starting$E     <- matrix(0L, p, p)      # empty graph
starting$alpha <- matrix(0, p, p)
starting$beta  <- matrix(0, p, p)
starting$delta <- rep(-10, p)
starting$gamma <- rep(0.5, p)
starting$psi   <- rep(0.5, p)
starting$tau   <- c(1, 1, 1, 1)
starting$rho   <- 0.01

## Standard deviations of the Metropolis proposals (BayesDAG0 worked example).
tuning <- list()
tuning$E     <- c(0.05, 0.05, 1.2, 0.3)
tuning$E_rev <- c(0.5, 0.5, 1.2, 0.3)
tuning$alpha <- 1.2
tuning$beta  <- 0.4
tuning$delta <- 1.2
tuning$gamma <- 0.3
tuning$psi   <- 0.8

## ---------------------------------------------------------------------------
## Priors. psi: Gamma(1, 2) (shape, rate), as for SPARK; tau (the precisions of
## DAG0's hierarchical effect priors) and rho: as in the BayesDAG0 worked example.
## ---------------------------------------------------------------------------
priors <- list()
priors$psi <- c(1, 2)
priors$tau <- c(1, 1)
priors$rho <- c(1, 1)

out <- mcmc_1sampleDAG0(dat, starting, tuning, priors,
                        n_sample    = N_SAMPLE,
                        n_burnin    = N_BURNIN,
                        n_chain     = N_CHAIN,
                        prob_swap   = PROB_SWAP,
                        temperature = TEMPERATURE,
                        do_par      = TRUE,
                        n_core      = N_CHAIN)

## Metropolis acceptance percents for alpha, beta, delta, gamma, psi.
cat("\n--- acceptance ---\n")
print(out$acceptance)

## ---------------------------------------------------------------------------
## PIPs: the share of retained cold-chain draws containing each edge.
##
## out$samples$E is a (p, p, n_sample - n_burnin) array. DAG0 stores E[j, k] for
## the edge k -> j, the TRANSPOSE of the convention of A_<i>.csv and the prior
## matrix, so each draw is transposed while averaging.
## ---------------------------------------------------------------------------
draws <- out$samples$E
if (!is.array(draws) || length(dim(draws)) != 3L)
  stop("expected out$samples$E to be a (p, p, n_draw) array, got: ",
       paste(dim(draws), collapse = "x"))
n_draw <- dim(draws)[3]
if (n_draw != N_SAMPLE - N_BURNIN)
  stop(sprintf("expected %d retained draws (n_sample - n_burnin) but got %d",
               N_SAMPLE - N_BURNIN, n_draw))

## c(2, 1) transposes while averaging: post[i, j] = P(edge i -> j).
post <- apply(draws != 0, c(2, 1), mean)
node_names <- colnames(dat)
dimnames(post) <- list(node_names, node_names)
write.csv(post, paste0(out_pref, "_post.csv"))
cat(sprintf("%d retained draws averaged; wrote %s_post.csv\n", n_draw, out_pref))
