#!/bin/bash 

unset PYTHONHOME

#/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/emceeInference/injections/createInjection -f $1

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/emceeInference/Driver.py -f $1
