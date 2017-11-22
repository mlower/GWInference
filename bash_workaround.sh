#!/bin/bash 

OUTPUTFILE=/home/marcus.lower/GWInference/logs/output.log
echo "Job number $@" >> $OUTPUTFILE

if [ "$@" == 0 ]
    then exit
fi

unset PYTHONHOME

job="150914_"$1


/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/GWInference/createInjection.py -f $job

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/GWInference/Driver.py -f $job

/home/marcus.lower/src/pycbc/bin/python /home/marcus.lower/GWInference/combineResults.py -f $job

echo "Job finished!" >> $OUTPUTFILE
