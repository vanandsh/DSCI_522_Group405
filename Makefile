all: doc/final_report.md

data/raw_data.csv : src/data_download.py
	python src/data_download.py --url="https://raw.githubusercontent.com/vanandsh/datasets/master/AB_NYC_2019.csv" --file_location="./data/raw_data.csv"

data/train.csv \
data/test.csv : \
data/raw_data.csv src/wrangle.py
	python src/wrangle.py --source_file_location="./data/raw_data.csv" --target_location="./data"
	

results/tables/summary_table.csv \
results/plots/categorical-plots.png \
results/plots/corr-plot.png \
results/plots/price-dist.png : data/train.csv data/test.csv src/eda_script.R
	Rscript src/eda_script.R --source_file="./data/train.csv" --target_location="./results"


results/tables/feature_importance_table.csv \
results/tables/mean_absolute_error_table.csv \
results/plots/ensemble_residual_distribution.png \
results/plots/ensemble_residual_plot.png : data/train.csv data/test.csv src/model.py
	python src/model.py --source_file_location="./data" --target_location="./results"
	

doc/final_report.md : \
results/tables/feature_importance_table.csv \
results/tables/mean_absolute_error_table.csv \
results/plots/ensemble_residual_distribution.png \
results/tables/summary_table.csv \
results/plots/categorical-plots.png \
results/plots/corr-plot.png 
	Rscript src/knit_rmd.R --source_file="./doc/final_report.Rmd"
	
clean:
	rm -f data/*
	rm -f results/plots/*
	rm -f results/tables/*
	rm -f doc/final_report.md doc/final_report.html