PDOC = python -m pdoc
PDOC_FLAGS = -d numpy --footer-text "missiontools v0.1.0"
DOCS_DIR = docs

.PHONY: docs docs-serve clean-docs

## Generate HTML documentation into docs/
docs:
	$(PDOC) missiontools $(PDOC_FLAGS) -o $(DOCS_DIR)/

## Live-preview documentation in the browser
docs-serve:
	$(PDOC) missiontools $(PDOC_FLAGS)

## Remove generated documentation
clean-docs:
	rm -rf $(DOCS_DIR)/
