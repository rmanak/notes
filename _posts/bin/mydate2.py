#!/usr/bin/env python

from __future__ import print_function

from datetime import datetime

import os.path

import sys


def parse_textfile_todict(fname):
    fdict = {}
    try:
        with open(fname,"r") as f:
            for line in f:
                a, b = line.split()
                fdict[a] = b
    except:
        fdict = {}
    return fdict

try:
    all_dates  = parse_textfile_todict('post_dates.param')
    fname = sys.argv[1]
    lm = os.path.getmtime(fname)
    lm = datetime.fromtimestamp(lm)
    print(all_dates.get(fname,lm.strftime('%Y-%m-%d')))

except: 
    print('')



