#!/bin/bash

# declare -a TRAIN_SETS=("bai" "chinese_2004" "japanese" "ob_ugrian" "an_train" "iel_train")
declare -a TEST_SETS=("aa" "an_test" "bah" "chinese_1964" "huon" "ie_test" "pn" "rom" "st" "tujia" "ura")
split=$1
#for file in ${TRAIN_SETS[@]};
#do
#    python manage.py prepare $file
#done

for file in ${TEST_SETS[@]};
do
    python manage.py prepare "${file}_prop_50_${split}_train"
    python manage.py prepare "${file}_prop_50_${split}_test"

done