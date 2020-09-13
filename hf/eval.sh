for TASK in r er rs ers
do
    echo -e "task: $TASK\n"
    for METHOD in nist bleu meteor
    do
        python $METHOD.py data/dd_dial_ne/test.target bart/pred/$TASK.txt
        echo
    done
    for METHOD in ent dist avg_len
    do
        python $METHOD.py bart/pred/$TASK.txt
        echo
    done
done
