#!/bin/bash

file_input=$1
file=${file_input%%.*}
file_output=${file}_data.csv

# extract the relevant data
grep Dot $file_input | cut -f 3 > ${file}_dot.txt
grep Matrix-Vector $file_input | cut -f 3 > ${file}_mv.txt
grep Matrix-Matrix $file_input | cut -f 3 > ${file}_mm.txt
grep Cholesky $file_input | cut -f 3 > ${file}_chol.txt
grep LU $file_input | cut -f 3 > ${file}_lu.txt
grep Inverse $file_input | cut -f 3 > ${file}_inv.txt

# combine the data column-wise
paste N.txt ${file}_dot.txt ${file}_mv.txt ${file}_mm.txt ${file}_chol.txt ${file}_lu.txt ${file}_inv.txt > ${file}_combine.txt

#add titles to the data
cat titles.txt ${file}_combine.txt > ${file_output}.tsv

# Convert from tab separated to CSV
tr -d '#' < ${file_output}.tsv | tr -s '\t' | tr '\t' ',' > ${file_output}

#cleanup
rm ${file}_{dot,mv,mm,chol,lu,inv,combine}.txt ${file_output}.tsv
