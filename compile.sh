#!/bin/bash
javac -cp ":weka-3-8-1/weka.jar" CSV2Arff.java
javac -cp ":weka-3-8-1/weka.jar" -cp ":autoweka_files/autoweka.jar" runAW2.java
