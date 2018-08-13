library(ggplot2)
library(RcppCNPy)



pops <- npyLoad("/Users/Neythen/masters_project/results/Q_learning_results/action_value/WORKING_data/Qpops.npy")
time <- matrix(seq(from = 0, to = 1000))

data <- cbind(time, pops)

data <- data.frame(time = time, P1 = pops[,1], P2 = pops[,2])

plot <- ggplot() +
geom_line(data = data, aes(x = time, y = P1), colour = "blue") +
geom_line(data = data, aes(x = time, y = P2), colour = "orange") +
scale_colour_manual(values = c("P1", "P2")) +
xlim(0,1000) +
ylim(0, 2) +
ylab("Population") +
xlab("Timestep")



ggsave("plot.png", plot,width = 10, height = 5, dpi = 120)



plot_pops <- function(loadpath, tmax) {

  pops <- npyLoad(loadpath)
  time <- matrix(seq(from = 0, to = tmax))

  data <- cbind(time, pops)

  data <- data.frame(time = time, P1 = pops[,1], P2 = pops[,2])

  plot <- ggplot() +
  geom_line(data = data, aes(x = time, y = P1), colour = "blue") +
  geom_line(data = data, aes(x = time, y = P2), colour = "orange") +
  scale_colour_manual(values = c("P1", "P2")) +
  xlim(0,1000) +
  ylim(0, 2) +
  ylab("Population") +
  xlab("Timestep")

  return(plot)

}

plot_survival <- function(loadpath, num_episodes, tmax, test_freq){
  survival_times <- matrix(npyLoad(loadpath))
  episodes <- matrix(seq(from = test_freq, to = num_episodes , by = test_freq))


  data <- cbind(episodes, survival_times)


  data <- data.frame(episodes = episodes, survival_times = survival_times)


  plot <- ggplot() +
  geom_line(data = data, aes(x = episodes, y = survival_times), colour = "blue") +

  xlim(0,num_episodes) +
  ylim(0, tmax) +
  ylab("Average Times Steps Survived") +
  xlab("Episode")

  return(plot)
}

plot_rewards <- function(loadpath, num_episodes, tmax, test_freq) {
  rewards <- matrix(npyLoad(loadpath))
  episodes <- matrix(seq(from = test_freq, to = num_episodes-1, by = test_freq))
  print(dim(rewards))
  print(dim(episodes))
  data <- cbind(episodes, rewards)

  data <- data.frame(episodes = episodes, survival_times = rewards)
  print(data)

  plot <- ggplot() +
  geom_line(data = data, aes(x = episodes, y = rewards), colour = "blue") +

  xlim(0,num_episodes) +
  ylim(-tmax, tmax) +
  ylab("Average Reward Recieved Per Episode") +
  xlab("Episode")

  return(plot)
}

pops_plot <- plot_pops("/Users/Neythen/masters_project/results/Q_learning_results/action_value/WORKING_data/Qpops.npy", 1000)
survival_plot <- plot_survival("/Users/Neythen/masters_project/results/recurrent_Q_results/action_value/WORKING_data/Q_train_survival.npy", 100, 1000, 5)
rewards_plot <- plot_rewards("/Users/Neythen/masters_project/results/Q_learning_results/action_value/WORKING_data/Qtest_rewards.npy", 100, 1000, 5)


ggsave("pops_plot.png", pops_plot,width = 10, height = 5, dpi = 120)
ggsave("surv_plot.png", survival_plot,width = 10, height = 5, dpi = 120)
ggsave("rew_plot.png", rewards_plot,width = 10, height = 5, dpi = 120)
