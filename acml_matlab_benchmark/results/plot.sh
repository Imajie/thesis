#!/bin/bash

echo 'set xlabel "N"
set log y
set ylabel "Time(s)"
set title "'"$2"'"
set datafile separator ","
set yrange [0.1 : 1e7 ]

plot "'"$1"'" every ::1 using 1:2 with lines title "Dot Product", \
	 "'"$1"'" every ::1 using 1:3 with lines title "Matrix-Vector", \
	 "'"$1"'" every ::1 using 1:4 with lines title "Matrix-Matrix", \
	 "'"$1"'" every ::1 using 1:5 with lines title "Cholesky", \
	 "'"$1"'" every ::1 using 1:6 with lines title "LU", \
	 "'"$1"'" every ::1 using 1:7 with lines title "Inverse";' | gnuplot --persist

