# author: Ofer Mansour, Jacky Ho, Anand Vemparala
# date: 2020-01-23

"This script runs eda to create plots and tables. 

Usage: eda_script.R --source_file=<source_file> --target_location=<target_location> 

Options:
--source_file=<source_file>     Takes a path/filename pointing to the data
--target_location=<target_location> Takes a path/filename prefix where to write the figure to and what to call it (*e.g.,* `results/this_eda.png`)


" -> doc

library(docopt)
library(tidyverse)
library(reshape2)
library(gridExtra)

main <- function(input, output) {
  
  # test to check that csv is inputted 
  if (substr(input, (nchar(input)+1)-3 ,nchar(input)) != "csv"){
    stop("Must input an csv file")
  }
  training_data  <- read_csv(input)
  
  # Correlation plot
  cors  <- training_data  %>%
    select(-neighbourhood_group, -neighbourhood, -room_type, -latitude, -longitude )  %>% 
    cor()
  
  get_lower_tri<-function(cormat){
    cormat[upper.tri(cormat)] <- NA
    return(cormat)
  }
  # Get upper triangle of the correlation matrix
  get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
  }
  
  upper_tri  <- get_upper_tri(cors)
  cors  <- melted_cormat <- melt(upper_tri, na.rm = TRUE)
  
  corr_plot  <- ggplot(data = cors, aes(Var2, Var1, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                         midpoint = 0, limit = c(-1,1), space = "Lab", 
                         name="Pearson\nCorrelation") +
    theme_minimal()+ 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                     size = 12, hjust = 1))+
    coord_fixed() +
    labs(x = "", y = "")

  # save plot (change file path/name if necessary)
  # ggsave('results/plots/01_corr-plot.png', plot = corr_plot)
  save_to = paste0(output,"/plots/corr-plot.png")
  ggsave(save_to, plot = corr_plot)
  
  # price distribution
  price_dist  <- ggplot(training_data, aes(x = factor(0), y = price)) +
    scale_y_log10() +
    geom_boxplot() +
    labs(x = "", y = "Price USD (log scaled)", 
         title = "Distribution of Price per Night in USD") +
    theme(axis.ticks.x= element_blank(), 
          axis.text.x=element_blank(),
          plot.title = element_text(hjust = 0.5))
  
  # save plot
  save_to = paste0(output,"/plots/price-dist.png")
  ggsave(save_to, plot = price_dist)
  
  ##### Categorical Plots ##########
  room_price  <- training_data  %>% 
    group_by(room_type)  %>% 
    summarize(mean_price = mean(price))  %>% 
    ggplot(., aes(x= reorder(room_type, -mean_price), y = mean_price)) +
    geom_bar(stat = 'identity', fill = 'rosybrown') +
    labs(x = 'Room type', y = 'Dollars (USD)', title = 'Mean Price per Night for each Room Type') + 
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12))
  
  neighb_price <- training_data  %>% 
    group_by(neighbourhood_group)  %>% 
    summarize(mean_price = mean(price))  %>% 
    ggplot(. , aes(x = reorder(neighbourhood_group, -mean_price), y = mean_price)) +
    geom_bar(stat = 'identity', fill = 'steelblue') +
    labs(x = 'Borough', y = 'Dollars (USD)', title = 'Mean Price per Night per Borough') + 
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12))
  
  
  # arranges plot in 1X2 grid
  categorical_plots <- arrangeGrob(room_price, neighb_price, ncol=2)
  # save plot
  
  save_to = paste0(output,"/plots/categorical-plots.png")
  ggsave(save_to, plot = categorical_plots, width = 9, height = 6)
  
}
opt <- docopt(doc)
main(opt[['--source_file']], opt[['--target_location']])