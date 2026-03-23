.PHONY: deps eda cv full_train plots predict all clean

deps:
	Rscript src/install_packages.R

eda:
	Rscript analyze_distributions.R
	Rscript scripts/01_eda.R

cv:
	Rscript scripts/02_cv.R

full_train:
	Rscript scripts/03_full_train.R

plots:
	Rscript scripts/04_plots.R

predict:
	Rscript scripts/05_predict_test.R

all: eda cv full_train plots

clean:
	rm -f outputs/results/*.csv outputs/results/*.png
	rm -f outputs/models/*.rds
