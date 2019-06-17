#!/bin/bash
s=''
space=" "
for i in $@; do
    c=$c$space$i
done

~/.virtualenvs/cv/bin/python $c
