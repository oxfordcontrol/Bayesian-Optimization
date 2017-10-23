`qEI_caller` <- function(x, x_all, y_all, covtype, cov_param, cov_var, var){
  m <- create_model(x_all, y_all, covtype, cov_param, cov_var, var)

  ei = qEI(x, m, type="SK")
  return(ei)
}

`qEI_grad_caller` <- function(x, x_all, y_all, covtype, cov_param, cov_var, var){
  m <- create_model(x_all, y_all, covtype, cov_param, cov_var, var)

  grad = qEI.grad(x, m, type="SK")
  return(grad)
}

`qEI_timing` <- function(x, x_all, y_all, covtype, cov_param, cov_var, var){
  m <- create_model(x_all, y_all, covtype, cov_param, cov_var, var)
  # Query ei first to avoid potential initialization timings
  ei = qEI(x, m, type="SK")

  timing <- system.time({
    ei = qEI(x, m, type="SK")
    grad = qEI.grad(x, m, type="SK")
  })

  # Return elapsed time
  return(timing[3])
}

`model_mean` <- function(x, x_all, y_all, covtype, cov_param, cov_var, var){
  m <- create_model(x_all, y_all, covtype, cov_param, cov_var, var)

  output <- predict(object=m, newdata=data.frame(x), type="SK")
  return(output$mean)
}

`model_var` <- function(x, x_all, y_all, covtype, cov_param, cov_var, var){
  m <- create_model(x_all, y_all, covtype, cov_param, cov_var, var)

  output <- predict(object=m, newdata=data.frame(x), type="SK")
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

  m <- km(design=data.frame(x_all),
          response=data.frame(y_all),
          coef.trend=0,
          coef.cov=c(cov_param),
          coef.var=c(cov_var),
          noise.var=rep(var,1,nrow(x_all)),
          covtype=covtype
  )

  return(m)
}