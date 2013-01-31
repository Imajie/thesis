#!/bin/bash

javac model_2012.java

for file in params/*_295.txt 
do

	echo "=============================="
	echo $file
	java model_2012 < $file
	echo "=============================="

done
