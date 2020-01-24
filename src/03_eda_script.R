library(tidyverse)
library(GGally)
library(gridExtra)
library(ggthemes)
theme_set(theme_minimal())

# change to input file
training_data  <- read_csv("training_data.csv")

# Correlation plot
corr_plot  <- training_data  %>% 
  select(-X1, -neighbourhood_group, -neighbourhood, -room_type, -latitude, -longitude )  %>%
  rename(no_host_listgs = calculated_host_listings_count )  %>% 
  ggpairs() +
  theme(strip.text.x = element_text(size = 9),
        strip.text.y = element_text(size = 9),
        axis.text.x = element_text(angle=90))

# save plot (change file path/name if necessary)
ggsave('results/plots/01_corr-plot.png', plot = corr_plot)

# price distribution
price_dist  <- ggplot(training_data, aes(x = price)) + 
  geom_histogram(bins = 500, fill = 'mediumpurple') +
  labs(x = 'Price per Night (USD)', y = 'Count', title =  'Distribution of Price per Night in USD') +
  theme(plot.title = element_text(hjust = 0.5))
# save plot
ggsave('results/plots/02_price-dist.png', plot = price_dist, width = 9, height = 6)


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
ggsave("results/plots/03_categorical-plots.png", plot = categorical_plots, width = 9, height = 6)


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

write.csv(summary_table, file = 'results/plots/01_summary-table.csv',  row.names = FALSE)