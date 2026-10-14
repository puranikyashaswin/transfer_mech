PYTHON ?= python3

.PHONY: verify test stats figures

verify:  ## recompute headline numbers from raw logs (stdlib only)
	$(PYTHON) scripts/verify_headline.py

test:    ## regression + harness unit tests
	$(PYTHON) -m pytest -q

stats:   ## full exploratory + confirmatory statistics (needs scipy)
	$(PYTHON) paper/make_tables.py
	$(PYTHON) paper/make_stats.py
	$(PYTHON) paper/make_stats_confirmatory.py
	cd results/confirmatory_L17 && $(PYTHON) stats_L17.py

figures: ## rebuild paper/figures (needs matplotlib)
	$(PYTHON) paper/make_figures.py
