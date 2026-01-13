.IGNORE:

SHELL = /bin/bash

MD = ./bin/run_markdown.py
PP = ./bin/preppage.py
INCF = ./bin/inc_file.py
TOCGEN = ./bin/pyhat.py

PYTHON_EXECUTABLE := $(shell command -v python3 || command -v python || where python)

# Check if Python executable was found, otherwise stop the make process
ifndef PYTHON_EXECUTABLE
$(error Python executable not found. Please install Python or add it to your PATH.)
endif

SOURCES = Template.txt test_template.txt

pagefiles=$(SOURCES:.txt=.html)

default: all

all: $(pagefiles)

%.html : %.txt template
	@$(MD) $*.txt > $*.w
	$(PP) template $*.w | $(INCF) > $*.html
	@grep 'table_of_contents' $*.html && $(PYTHON_EXECUTABLE) $(TOCGEN) -d. $*.html 2>&1 > /dev/null || echo ""
	@echo $*.html created
	/bin/cp $*.html `./bin/mydate2.py $*.txt`-$*.html
	/bin/rm -f $*.w $*.html

clean:
	@for fname in $(pagefiles); do \
		/bin/rm -rf *$$fname ; \
	done

