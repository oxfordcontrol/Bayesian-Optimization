# Code demonstrating DiceOptim's qEI outputing random values

library('DiceOptim')

x_all <- replicate(2, rnorm(100))
y_all <- matrix(0, nrow=100, ncol=1)
m <- km(design=data.frame(x_all),
        response=data.frame(y_all),
        coef.trend=0,
        coef.cov=c(.1, .1),
        coef.var=c(1),
        noise.var=rep(1e-6, 100),
        covtype='gauss'
)

x <- replicate(2, rnorm(6))

ei <- qEI(x, m, type="SK")
print(ei)
