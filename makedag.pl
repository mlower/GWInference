#! /usr/bin/perl

$home = "/home/marcus.lower";
$sub = "condor";
$njobs = 25;

open(dag,">./run_Inference.dag");
for ($i=0; $i<=$njobs; $i=$i+1) {
    print dag "JOB $i /home/marcus.lower/public_html/projects/emceeSURF/emceeInference/$sub.sub\n";
    print dag "VARS $i ";
    print dag "jobNumber=\"$i\" ";
    print dag "\n\n";
}
print dag "PARENT 0 CHILD ";
for ($j=1; $j<$i; $j=$j+1) {print dag "$j ";}
print dag "\n";
close(dag);

