for z in logs/*; do
    for b in $z/*; do
        rm -rf $b/iterations
        mkdir $b/iterations
        for out in $b/outs/*.out; do
            filename="$(basename "$out")"
            stem=${filename%.*}
            echo $stem
            grep -a -i "iteration  \+" $out > $b/iterations/$stem.iterations
        done
    done
done