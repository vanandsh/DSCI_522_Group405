# author: Ofer Mansour, Jacky Ho, Anand Vemparala
# date: 2020-01-23

"This script prints out docopt args.

Usage: eda_script.R --source_file=<source_file> --target_location=<target_location> 

Options:
--source_file=<source_file>     Takes a path/filename pointing to the data
--target_location=<target_location> Takes a path/filename prefix where to write the figure to and what to call it (*e.g.,* `results/this_eda.png`)


" -> doc

library(docopt)
library(tidyverse)
library(GGally)
library(gridExtra)

main <- function(input, output) {
  
  training_data  <- read_csv(input)
  
  # Correlation plot
  corr_plot  <- training_data  %>%
    select(-neighbourhood_group, -neighbourhood, -room_type, -latitude, -longitude )  %>%
    rename(no_host_listgs = calculated_host_listings_count )  %>%
    ggpairs() +
    theme(strip.text.x = element_text(size = 8),
          strip.text.y = element_text(size = 8),
          axis.text.x = element_text(angle=90))

  # save plot (change file path/name if necessary)
  # ggsave('results/plots/01_corr-plot.png', plot = corr_plot)
  save_to = paste0(output,"/plots/corr-plot.png")
  ggsave(save_to, plot = corr_plot)
  
  # price distribution
  price_dist  <- ggplot(training_data, aes(x = price)) + 
    geom_histogram(bins = 500, fill = 'mediumpurple') +
    labs(x = 'Price per Night (USD)', y = 'Count', title =  'Distribution of Price per Night in USD') +
    theme(plot.title = element_text(hjust = 0.5))
  
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
    theme(plot.title = element_text(hjust = 0.5, size = 12),
          axis.text.x = element_text(size = 7),
          axis.text.y = element_text(size = 7),
          axis.title.x = element_text(size = 9),
          axis.title.y = element_text(size = 9))
  
  neighb_price <- training_data  %>% 
    group_by(neighbourhood_group)  %>% 
    summarize(mean_price = mean(price))  %>% 
    ggplot(. , aes(x = reorder(neighbourhood_group, -mean_price), y = mean_price)) +
    geom_bar(stat = 'identity', fill = 'steelblue') +
    labs(x = 'Neighbourhood Group', y = 'Dollars (USD)', title = 'Mean Price per Night of Listings per Neighbourhood Group') + 
    theme(plot.title = element_text(hjust = 0.5, size = 9),
          axis.text.x = element_text(size = 7),
          axis.text.y = element_text(size = 7),
          axis.title.x = element_text(size = 9),
          axis.title.y = element_text(size = 9))
  
  
  # arranges plot in 1X2 grid
  categorical_plots <- arrangeGrob(room_price, neighb_price, ncol=2)
  # save plot
  
  save_to = paste0(output,"/plots/categorical-plots.png")
  ggsave(save_to, plot = categorical_plots, width = 9, height = 6)
  
  # Summary table for neighb group, room and price
  summary_table  <- training_data  %>% 
    group_by(neighbourhood_group, room_type)  %>% 
    summarize(Listings_Count = n(),
              Mean_Price = round(mean(price),2),
              Max_Price = max(price))   %>% 
    rename(`Neighbourhood Group` = neighbourhood_group, 
           `Room Type` = room_type, 
           `Number of Listings` = Listings_Count,
           `Mean Price per Night` = Mean_Price,
           `Max Price per Night` = Max_Price)
  
  
  save_to = paste0(output,"/tables/summary-table.csv")
  write.csv(summary_table, file = save_to,  row.names = FALSE)

}

opt <- docopt(doc)
main(opt[['--source_file']], opt[['--target_location']])