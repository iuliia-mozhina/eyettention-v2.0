

x_m <- 2
x_sd <- 1

dat <- expand.grid(x=seq(-1, 5, length.out = 1000)) %>% mutate(
  Gamma = dgamma(x, shape = x_m^2 / x_sd^2, rate = x_m / x_sd^2),
  Gauss = dnorm(x, mean = x_m, sd = x_sd)
) %>% gather(dist, y, -x)

p13 <- ggplot(dat) +
  theme_apa() + theme(legend.position = "bottom") + color_apa() +
  geom_line(aes(x=x,y=y,color=dist,linetype=dist,group=dist)) +
  geom_vline(xintercept = x_m, color="red") +
  geom_vline(xintercept = 0, color="black") +
  labs(color=NULL, linetype=NULL, x="Saccade amplitude", y="Probability density")

save_fig_narrow("figs-33/fig13-gamma-vs-gauss-dist.pdf", p13, aspect=1)
