# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = ../../docs/pyldpc-docs/
PDFBUILDDIR   = /tmp
PDF           = ../manual.pdf


.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html-noplot to make standalone HTML files, without plotting anything"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo "  coverage   to run coverage check of the documentation (if enabled)"
	@echo "  install    to make the html and push it online"

.PHONY: clean

clean:
	rm -rf $(BUILDDIR)html/auto_examples
	rm -rf $(BUILDDIR)html/generated
	rm -rf $(BUILDDIR)html/_modules
	rm -rf $(BUILDDIR)html/_images
	rm -rf auto_examples/*
	rm -rf generated/*
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


latexpdf:
	@$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PDFBUILDDIR)/latex
	#                                          ^^^
	@echo "Running LaTeX files through pdflatex..."
	make -C $(PDFBUILDDIR)/latex all-pdf
	#         ^^^
	cp $(PDFBUILDDIR)/latex/*.pdf $(PDF)
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	@echo "pdflatex finished; see $(PDF)"


commithtml: html
	cd $(BUILDDIR)html; git add . ; git commit -m "rebuilt docs"; git push origin gh-pages

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
