#!/bin/bash

java -cp ":../weka-3-8-1/weka.jar" CSV2Arff data/samples_unt.txt data/samples_unt.arff

java -cp ":../weka-3-8-1/weka.jar" CSV2Arff data/samples_wavelet1ap.txt data/samples_wavelet1ap.arff
java -cp ":../weka-3-8-1/weka.jar"  CSV2Arff data/samples_wavelet1dl.txt data/samples_wavelet1dl.arff

java -cp ":../weka-3-8-1/weka.jar"  CSV2Arff data/samples_wavelet2ap.txt data/samples_wavelet2ap.arff
java -cp ":../weka-3-8-1/weka.jar"  CSV2Arff data/samples_wavelet2dl.txt data/samples_wavelet2dl.arff

java -cp ":../weka-3-8-1/weka.jar"  CSV2Arff data/samples_wavelet3ap.txt data/samples_wavelet3ap.arff
java -cp ":../weka-3-8-1/weka.jar"  CSV2Arff data/samples_wavelet3dl.txt data/samples_wavelet3dl.arff

java -cp ":../weka-3-8-1/weka.jar"  CSV2Arff data/samples_wavelet4ap.txt data/samples_wavelet4ap.arff
java -cp ":../weka-3-8-1/weka.jar"  CSV2Arff data/samples_wavelet4dl.txt data/samples_wavelet4dl.arff

