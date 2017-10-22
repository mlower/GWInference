#!/bin/bash 

OUTPUTFILE=/home/marcus.lower/public_html/projects/emceeSURF/GWInference/logs/output.log
echo "Job number $@" >> $OUTPUTFILE

if [ "$@" == 0 ]
    then exit
fi

unset PYTHONHOME

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/GWInference/injections/createInjection.py -f $1

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/GWInference/Driver.py -f $1

echo "Job finished!" >> $OUTPUTFILE
