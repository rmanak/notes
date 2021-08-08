.IGNORE:

SHELL = /bin/bash

MD = ./bin/Markdown.pl
#MD = cat
PP = ./bin/preppage.pl
INCF = ./bin/inc_file.pl
TOCGEN = ./bin/pyhat.py


SOURCES = Template.txt test_template.txt

pagefiles=$(SOURCES:.txt=.html)

default: all

all: $(pagefiles)

%.html : %.txt template
	@$(MD) $*.txt > $*.w
	$(PP) template $*.w | $(INCF) > $*.html
	@grep 'table_of_contents' $*.html && python2 $(TOCGEN) -d. $*.html 2>&1 > /dev/null || echo ""
	@echo $*.html created
	/bin/cp $*.html `./bin/mydate2.py $*.txt`-$*.html
	/bin/rm -f $*.w $*.html

clean:
	@for fname in $(pagefiles); do \
		/bin/rm -rf *$$fname ; \
	done


