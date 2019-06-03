#!/bin/sh
#$ -S /bin/bash
#$ -v PATH=/home/data/webcomp/RAMMCAP-ann/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#$ -v BLASTMAT=/home/data/webcomp/RAMMCAP-ann/blast/bin/data
#$ -v LD_LIBRARY_PATH=/home/data/webcomp/RAMMCAP-ann/gnuplot-install/lib
#$ -v PERL5LIB=/home/hying/programs/Perl_Lib
#$ -q cdhit_webserver.q
#$ -pe orte 4
#$ -l h_rt=24:00:00


#$ -e /data5/data/webcomp/web-session/1559564645/1559564645.err
#$ -o /data5/data/webcomp/web-session/1559564645/1559564645.out
cd /data5/data/webcomp/web-session/1559564645
sed -i "s/\x0d/\n/g" 1559564645.fas.db1
sed -i "s/\x0d/\n/g" 1559564645.fas.db2

faa_stat.pl 1559564645.fas.db1
faa_stat.pl 1559564645.fas.db2
/data5/data/NGS-ann-project/apps/cd-hit/cd-hit-2d -i 1559564645.fas.db1 -i2 1559564645.fas.db2 -o 1559564645.fas.db2novel -c 0.4 -n 2  -G 1 -g 1 -b 20 -l 10 -s 0.0 -aL 0.0 -aS 0.0 -s2 1.0 -S2 0 -T 4 -M 32000
/data5/data/NGS-ann-project/apps/cd-hit/clstr_sort_by.pl no < 1559564645.fas.db2novel.clstr > 1559564645.fas.db2novel.clstr.sorted
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list.pl 1559564645.fas.db2novel.clstr 1559564645.clstr.dump
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1559564645.clstr.dump 1559564645.clstr_no.dump
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1559564645.clstr.dump 1559564645.clstr_len.dump len
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1559564645.clstr.dump 1559564645.clstr_des.dump des
gnuplot1.pl < 1559564645.fas.db2novel.clstr > 1559564645.fas.db2novel.clstr.1; gnuplot2.pl 1559564645.fas.db2novel.clstr.1 1559564645.fas.db2novel.clstr.1.png
tar -zcf 1559564645.result.tar.gz * --exclude=*.dump --exclude=*.env
echo hello > 1559564645.ok
