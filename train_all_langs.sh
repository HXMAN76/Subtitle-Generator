#!/bin/bash
# Submit separate jobs for each language
LANGS="hi ta te bn mr gu kn ml pa or as"
for LANG in $LANGS; do
    sbatch --export=TARGET_LANG=$LANG train_nmt.sh
done
