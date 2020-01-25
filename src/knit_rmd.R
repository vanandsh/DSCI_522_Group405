# author: Ofer Mansour, Jacky Ho, Anand Vemparala
# date: 2020-01-23

"This script knits Rmarkdowns.
Usage: eda_script.R --source_file=<source_file>
Options:
--source_file=<source_file>     Takes a path/filename pointing to the data
" -> doc

library(docopt)


main <- function(input) {
  
  # check that input file is an Rmd
  if (substr(input, (nchar(input)+1)-3 ,nchar(input)) != "Rmd"){
    stop("Must input an Rmd file")
  }
  rmarkdown::render(input, "github_document")
}

opt <- docopt(doc)
main(opt[['--source_file']])