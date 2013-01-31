#!/bin/bash

for file in results/*.txt; do
	# see if this file needs splitting
	if [ "`head -n 1 $file`" == "_NO_CACHE_" ]; then
		noCache=${file/_read_/_read_uncached_}
		cache=${file/_read_/_read_cached_}

		# split file starting at second line -> remove _NO_CACHE_ header
		tail -n+2 $file | awk -v RS="_WITH_CACHE_" '{ if (NR == 1) print $0 > "'$noCache'"; else print $0 > "'$cache'"}'

		# remove blank lines
		sed -i '/^$/d' $noCache
		sed -i '/^$/d' $cache

		# delete original
		rm $file
	fi
done
