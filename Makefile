.PHONY: deps eda cv plots predict all clean

deps:
	Rscript src/install_packages.R

eda:
	Rscript scripts/01_eda.R

cv:
	Rscript scripts/02_cv.R

plots:
	Rscript scripts/04_plots.R

predict:
	Rscript scripts/05_predict_test.R

all: eda cv plots

clean:
	rm -f outputs/results/*.csv outputs/results/*.png
	rm -f outputs/models/*.rds
