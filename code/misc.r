# generate a n by 2 matrix with random 0s and 1s
X <- c(0.02, 0.07, 0.15)
Y <- c(0.01, 0.04, 0.08)

plot(X, Y, type = "l", col = "blue", lwd = 2, xlab = "x", ylab = "y", main = "CDF of a Uniform Distribution")
model <- lm(Y ~ X)
print(summary(model))
