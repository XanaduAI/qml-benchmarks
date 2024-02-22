library(ECoL)

for (k in 2:20) {
  dim <- 10 # note: change to k if dimension is varied!
  path_in <- sprintf("../../benchmarks/hyperplanes_diff/hyperplanes-10d-from3d-%sn_train.csv", k)
  path_out <- sprintf("PROFILE-hyperplanes-10d-from3d-%sn_train.csv", k)
  n_cols <- dim + 1
  dataset <- read.csv(path_in, header=FALSE, col.names=paste0("C", 1:n_cols) )
  labels <- dataset[,paste0('C', n_cols)]
  X <- dataset[, 1:dim]
  result <- ECoL::complexity(X, labels)
  write.csv(result, file=path_out)
}
