#!/bin/bash
#SBATCH --hint=nomultithread        
#SBATCH --account=qgz@cpu
#SBATCH --time=10:00:00             
#SBATCH --qos=qos_cpu-t3

snapchots=(
    2014-15
    2014-23
    2014-35
    2014-41
    2014-42
    2014-49
    2014-52
    2015-14
    2015-22
    2015-27
    2015-32
    2015-35
    2015-40
    2015-48
    2016-07
    2016-18
    2016-22
    2016-26
    2016-30
    2016-36
    2016-40
    2016-44
    2016-50
    2017-04
    2017-09
    2017-17
    2017-22
    2017-26
    2017-30
    2017-34
    2017-39
    2017-43
    2017-47
    2017-51
    2018-05
    2018-09
    2018-13
    2018-17
    2018-22
    2018-26
    2018-30
    2018-34
    2018-39
    2018-43
    2018-47
    2018-51
    2019-04
    2019-09
    2019-13
    2019-18
    2019-22
    2019-26
    2019-30
    2019-35
    2019-39
    2019-43
    2019-47
    2019-51
    2020-05
    2020-10
    2020-16
    2020-24
    2020-29
    2020-34
    2020-40
    2020-45
    2020-50
    2021-04
    2021-10
    2021-17
    2021-21
    2021-25
    2021-31
    2021-39
    2021-43
    2021-49
    2022-05
    2022-21
    2022-27
    2022-33
    2022-40
    2022-49
    2023-06
    2023-14)

###############
### BASE PROCESSING
# snapchots_tail=("${snapchots[@]: -10}")
# for snapchot in "${snapchots_tail[@]}"
# do
#     ./base.sh $snapchot fr
#     ./base.sh $snapchot es
#     ./base.sh $snapchot de
#     ./base.sh $snapchot it
# done

# snapchots_head=("${snapchots[@]:0:${#snapchots[@]}-10}")
# for snapchot in "${snapchots_head[@]}"
# do
#     ./base.sh $snapchot fr
# done

###############
### URL FILTERING & PII
for snapchot in "${snapchots[@]}"
do
    ./domains.sh $snapchot fr
done

snapchots_tail=("${snapchots[@]: -10}")
for snapchot in "${snapchots_tail[@]}"
do
    ./domains.sh $snapchot es
    ./domains.sh $snapchot de
    ./domains.sh $snapchot it
done

###############
### MINHASH
# for snapchot in "${snapchots[@]}"
# do
#     ./minhash.sh $snapchot fr
# done

# snapchots_tail=("${snapchots[@]: -10}")
# for snapchot in "${snapchots_tail[@]}"
# do
#     ./minhash.sh $snapchot es
#     ./minhash.sh $snapchot de
#     ./minhash.sh $snapchot it
# done