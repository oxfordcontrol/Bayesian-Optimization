`qEI_caller` <- function(x, m) {
  ei = qEI(x, m, type="SK")
  return(ei)
}

`qEI_grad_caller` <- function(x, m) {
  grad = qEI.grad(x, m, type="SK")
  return(grad)
}

`model_mean` <- function(x, m) {
  design <- data.frame(x)
  names(design) <- colnames(m@X)
  output <- predict(object=m, newdata=design, type="SK")
  return(output$mean)
}

`model_var` <- function(x, m) {
  design <- data.frame(x)
  names(design) <- colnames(m@X)
  output <- predict(object=m, newdata=design, type="SK")
  return(output$sd^2)
}

`create_model` <- function(x_all, y_all, covtype, cov_param, cov_var, var){
  # The Gaussian kernel (RBF in GPy) is the only kernel that matches to the GPy
  # This is due to DiceKriging using l-1 distance
  # See "CovFuns.c" in DiceKriging source and
  # Journal of Statistical Software
  # DiceKriging, DiceOptim: Two R Packages for the Analysis of Computer
  # Experiments by Kriging-Based Metamodeling and Optimization

  stopifnot(covtype=='gauss')

  design <- data.frame(x_all)
  m <- km(design=design,
          response=data.frame(y_all),
          coef.trend=0,
          coef.cov=c(cov_param),
          coef.var=c(cov_var),
          noise.var=rep(var,1,nrow(x_all)),
          covtype=covtype
  )

  return(m)
}


`create_model_1d_quadratic_mean` <- function(x_all, y_all, covtype, cov_param, cov_var, var, trend){
  # Same as create_model but for 1d functions with quadratic mean function

  stopifnot(covtype=='gauss')

  design <- data.frame(x_all)
  # This also asserts that our data must be 1d
  names(design) <- c('x1')
  trend <- c(0, trend)
  m <- km(formula=~I(x1^2),
          design=design,
          response=data.frame(y_all),
          coef.trend=trend,
          coef.cov=c(cov_param),
          coef.var=c(cov_var),
          noise.var=rep(var,1,nrow(x_all)),
          covtype=covtype
  )

  return(m)
}
