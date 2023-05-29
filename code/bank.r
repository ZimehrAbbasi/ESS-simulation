# import banks data

d <- read.csv("C:\\Users\\Zimehr\\Desktop\\Senior\\Spring\\MATH 30.04\\Final Project\\ESS-simulation\\code\\banks.csv", header = TRUE, sep = ",")

#  remove rows with assets or liabilities less than 1
d <- d[d$A > 1,]
d <- d[d$L > 1,]

total.assets <- d$A * 1000
total.liabilities <- d$L * 1000

# plot total assets vs total liabilities


# log data
a <- log(total.assets)
l <- log(total.liabilities)

png("C:\\Users\\Zimehr\\Desktop\\Senior\\Spring\\MATH 30.04\\Final Project\\Final Paper\\images\\totAvstotL.png", width = 1500, height = 1500, res = 300)
par(mar=c(4,3,1,1))
plot(a, l, col="blue", xlab="Log Total Asset Value (in $)")
abline(lm(l ~ a), col="red")
mtext("Log Total Liability Value (in $)", side=2, line=-1, outer=TRUE, cex=1, font=1)
dev.off()

# run a regression
reg <- lm(l ~ a)
print(summary(reg))

# densities
png("C:\\Users\\Zimehr\\Desktop\\Senior\\Spring\\MATH 30.04\\Final Project\\Final Paper\\images\\totAtotL.png", width = 1500, height = 1500, res = 300)
par(mar=c(4,3,1,1))
plot(density(a), col="blue", xlab="Log Value (in $)", main="")
lines(density(l), col="red")
mtext("Density", side=2, line=-1, outer=TRUE, cex=1, font=1)
legend("topright", c("Total Assets", "Total Liabilities"), col=c("blue", "red"), lty=1)
dev.off()

asset_mean <- mean(a)
log_asset_mean <- log(mean(total.assets)^2/sqrt(mean(total.assets)^2 + var(total.assets)))
liability_mean <- mean(l)
asset_sd <- sd(a)
liability_sd <- sd(l)

print(asset_mean)
print(liability_mean)
print(asset_sd)
print(liability_sd)

# plot normal distribution

assets <- rnorm(mean = asset_mean, sd = asset_sd, n = 1000)
liabilities <- rnorm(mean = liability_mean, sd = liability_sd, n = 1000)

# plot(assets, liabilities, main = "Normal Distribution of Total Assets and Total Liabilities", xlab = "Total Assets (in $)", ylab = "Total Liabilities (in $)", col="blue")

# normal distribution
# curve(dnorm(x, mean = asset_mean, sd = asset_sd), col = "blue", lwd = 2, add = TRUE)
# curve(dnorm(x, mean = liability_mean, sd = liability_sd), col = "red", lwd = 2, add = TRUE)