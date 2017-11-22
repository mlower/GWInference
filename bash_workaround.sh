#!/bin/bash 

OUTPUTFILE=/home/marcus.lower/public_html/projects/emceeSURF/GWInference/logs/output.log
echo "Job number $@" >> $OUTPUTFILE

if [ "$@" == 0 ]
    then exit
fi

unset PYTHONHOME

job="170104_"$1


/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/GWInference/injections/createInjection.py -f $job

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/GWInference/Driver.py -f $job

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/public_html/projects/emceeSURF/GWInference/combineResults.py -f $job

echo "Job finished!" >> $OUTPUTFILE
