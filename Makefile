# AirBnB NewYork
# author: Ofer Mansour, Jacky Ho, Anand Vemparala
# date: 2020-01-30

# Runs complete analysis
all: doc/final_report.md

# download data
data/raw_data.csv : src/data_download.py
	python src/data_download.py --url="https://raw.githubusercontent.com/vanandsh/datasets/master/AB_NYC_2019.csv" --file_location="./data/raw_data.csv"


# Wrangle data & split into train and test after preprocessing
data/train.csv \
data/test.csv : \
data/raw_data.csv src/wrangle.py
	python src/wrangle.py --source_file_location="./data/raw_data.csv" --target_location="./data"
	
# Create EDA figures & tables
results/tables/summary_table.csv \
results/plots/categorical-plots.png \
results/plots/corr-plot.png \
results/plots/price-dist.png : data/train.csv data/test.csv src/eda_script.R
	Rscript src/eda_script.R --source_file="./data/train.csv" --target_location="./results"


# Run models and generate performance metrics and figures
results/tables/feature_importance_table.csv \
results/tables/mean_absolute_error_table.csv \
results/plots/model_result_charts.png : data/train.csv data/test.csv src/model.py
	python src/model.py --source_file_location="./data" --target_location="./results"

	
# Renders final report
doc/final_report.md : \
results/tables/feature_importance_table.csv \
results/tables/mean_absolute_error_table.csv \
results/plots/model_result_charts.png \
results/tables/summary_table.csv \
results/plots/categorical-plots.png \
results/plots/corr-plot.png 
	Rscript src/knit_rmd.R --source_file="./doc/final_report.Rmd"
	
# Clean up all intermediate and results files	
clean:
	rm -f data/*
	rm -f results/plots/*
	rm -f results/tables/*
	rm -f doc/final_report.md doc/final_report.html
