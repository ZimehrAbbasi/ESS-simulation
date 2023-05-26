# import banks data

d <- read.csv("banks.csv", header = TRUE, sep = ",")

#  remove rows with assets or liabilities less than 1
d <- d[d$Total.Assets > 1,]
d <- d[d$Total.Domestic.Deposits > 1,]

total.assets <- d$Total.Assets
total.liabilities <- d$Total.Domestic.Deposits

# log data
a <- log(total.assets)
l <- log(total.liabilities)

# densities
plot(density(a), main = "Density of Total Assets", xlab = "Total Assets", col="blue")
lines(density(l), col="red")
legend("topright", c("Total Assets", "Total Liabilities"), col=c("blue", "red"), lty=1)

asset_mean <- mean(a)
liability_mean <- mean(l)
asset_sd <- sd(a)
liability_sd <- sd(l)

print(asset_mean)
print(liability_mean)
print(asset_sd)
print(liability_sd)

# normal distribution
curve(dnorm(x, mean = asset_mean, sd = asset_sd), col = "blue", lwd = 2, add = TRUE)
curve(dnorm(x, mean = liability_mean, sd = liability_sd), col = "red", lwd = 2, add = TRUE)