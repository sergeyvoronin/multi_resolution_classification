#!/bin/bash

# check integrity of csv files by verifying column count for each row

# count cols in each row
perl -nle 's/".*?"//g;print s/,//g+1' data/samples_unt.csv  > cache/col_count1.txt
perl -nle 's/".*?"//g;print s/,//g+1' data/samples_wavelet1dl.csv   > cache/col_count2.txt

