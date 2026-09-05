# Documentation build.
#
# Everything runs through `uv run`, which resolves the docs extra from the
# repository's pyproject.toml. uv is not on PATH in a non-interactive shell
# unless mise has been activated; `mise exec -- make html` is the way in if
# `uv: command not found` is what you get.
#
#   make html        one build, warnings are errors
#   make live        rebuild and reload in a browser as you edit
#   make linkcheck   verify every external link
#   make clean       remove _build, including the Doxygen XML

UV          ?= uv
SPHINXOPTS  ?=
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: html live linkcheck clean

# -W --keep-going: fail on the first warning, but report all of them first, so
# one build tells you everything that is wrong instead of one thing at a time.
html:
	$(UV) run --extra docs sphinx-build -W --keep-going -b html \
		$(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

# conf.py runs Doxygen at import, and sphinx-autobuild re-imports it on every
# rebuild, so watching ../include regenerates the C++ XML as the headers
# change. ../python and ../examples are watched for the same reason: autodoc
# and the literalinclude blocks read them.
live:
	$(UV) run --extra docs sphinx-autobuild -b html \
		--watch ../include --watch ../python --watch ../examples \
		$(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

linkcheck:
	$(UV) run --extra docs sphinx-build -W --keep-going -b linkcheck \
		$(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
