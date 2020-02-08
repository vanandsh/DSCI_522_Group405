FROM continuumio/anaconda3 

# Install base R
RUN apt-get update && \
    apt-get install r-base r-base-dev -y && \
    apt-get install libcurl4-openssl-dev -y && \
    apt-get install libssl-dev -y && \
    apt-get install make -y
                           
RUN Rscript -e "install.packages('knitr')"
RUN Rscript -e "install.packages('docopt')"
RUN Rscript -e "install.packages('gridExtra')"
RUN Rscript -e "install.packages('reshape2')"
RUN Rscript -e "install.packages('kableExtra')"
RUN Rscript -e "install.packages('tidyverse')"
RUN Rscript -e "install.packages('png')"
RUN Rscript -e "install.packages('rmarkdown')"
    
# install python packages
RUN /opt/conda/bin/conda install -y -c anaconda docopt
RUN /opt/conda/bin/conda install -y -c anaconda requests
RUN /opt/conda/bin/conda install -y -c anaconda pandas
RUN /opt/conda/bin/conda install -y -c anaconda numpy
RUN /opt/conda/bin/conda install -y -c anaconda altair
RUN /opt/conda/bin/conda install -y -c anaconda selenium
RUN /opt/conda/bin/conda install -y -c anaconda scikit-learn
RUN /opt/conda/bin/conda install -y -c anaconda lightgbm
RUN /opt/conda/bin/conda install -y -c conda-forge xgboost
RUN pip install schema

# Install chromium and chromedriver
RUN apt install -y chromium && apt-get install -y libnss3 && apt-get install unzip

RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
    && unzip /tmp/chromedriver.zip -d /usr/bin/ \
    && rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver
    
# Put Anaconda Python in PATH
ENV PATH="/opt/conda/bin:${PATH}"
ENV PATH="/usr/bin/chromedriver:${PATH}"
CMD ["/bin/bash"]