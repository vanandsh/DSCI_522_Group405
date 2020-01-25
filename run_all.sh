# python src/data_download.py --url="https://raw.githubusercontent.com/vanandsh/datasets/master/AB_NYC_2019.csv" --file_location="./data/raw_data.csv"
# 
# python src/wrangle.py --source_file_location="./data/raw_data.csv" --target_location="./data"
# 
# Rscript.exe src/eda_script.R --source_file="./data/train.csv" --target_location="./results"
# 
python src/model.py --source_file_location="./data" --target_location="./results"