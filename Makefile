RSCRIPT := Rscript
INSTALL_PACKAGES := $(RSCRIPT) src/install_packages.R

.PHONY: deps eda cv plots predict all clean

deps:
	$(INSTALL_PACKAGES)

eda:
	$(INSTALL_PACKAGES)
	$(RSCRIPT) scripts/01_eda.R

cv:
	$(INSTALL_PACKAGES)
	$(RSCRIPT) scripts/02_cv.R

plots:
	$(INSTALL_PACKAGES)
	$(RSCRIPT) scripts/04_plots.R

predict:
	$(INSTALL_PACKAGES)
	$(RSCRIPT) scripts/05_predict_test.R

# Single install check, then EDA → CV → plots (same as separate makes, without repeating install)
all:
	$(INSTALL_PACKAGES)
	$(RSCRIPT) scripts/01_eda.R
	$(RSCRIPT) scripts/02_cv.R
	$(RSCRIPT) scripts/04_plots.R

clean:
	rm -f outputs/results/*.csv outputs/results/*.png
	rm -f outputs/models/*.rds
