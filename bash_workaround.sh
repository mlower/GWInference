#!/bin/bash 

unset PYTHONHOME

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/GWInference/injections/createInjection.py -f 1 #$1

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/GWInference/Driver.py -f 1 #$1
