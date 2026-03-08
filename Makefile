SPHINX_BUILD  = sphinx-build
SPHINX_AUTO   = sphinx-autobuild
SPHINX_SOURCE = docs/source
SPHINX_OUT    = docs/_build/html

.PHONY: docs docs-serve clean-docs

## Generate HTML documentation into docs/_build/html/
docs:
	$(SPHINX_BUILD) -b html $(SPHINX_SOURCE) $(SPHINX_OUT)

## Live-preview documentation in the browser (auto-rebuild on changes)
docs-serve:
	$(SPHINX_AUTO) $(SPHINX_SOURCE) $(SPHINX_OUT)

## Remove generated documentation
clean-docs:
	rm -rf docs/_build/
