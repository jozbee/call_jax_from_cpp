# Documentation build. Everything runs through `uv run`, which resolves the
# docs extra from pyproject.toml; `mise exec -- make html` is the way in when
# uv is not on PATH.
#
#   make html        one build, warnings are errors
#   make live        rebuild and reload in a browser as you edit
#   make linkcheck   verify every external link
#   make clean       remove _build, including the Doxygen XML

UV          ?= uv
SPHINXOPTS  ?=
SOURCEDIR    = .
BUILDDIR     = _build

# -W --keep-going: fail on any warning, but report all of them in one build.
SPHINXBUILD  = $(UV) run --extra docs sphinx-build -W --keep-going

.PHONY: html live linkcheck clean

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

# sphinx-autobuild re-imports conf.py, which runs Doxygen, so watching
# ../include regenerates the XML; autodoc and literalinclude read the other two.
live:
	$(UV) run --extra docs sphinx-autobuild -b html \
		--watch ../include --watch ../python --watch ../examples \
		$(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
