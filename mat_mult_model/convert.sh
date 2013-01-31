#!/bin/bash

sed 's/   /\t/g' $1 > $1.tabbed
cut -f 2,4 $1.tabbed > $1.cut
sed '/^$/d' $1.cut > $1.proc

rm $1.tabbed $1.cut

