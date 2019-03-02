#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22:20:31 2019

@author: dylan
"""

import json

json_file='tune_dump.json'
json_data=open(json_file)

parsed_json = json.load(json_data)

notes = ""

for tune in parsed_json:
    if tune['type'] == "reel":
        notes += " " + tune['abc']

textfile = open("rawnotes.txt", "w")
textfile.write(notes)
json_data.close()


