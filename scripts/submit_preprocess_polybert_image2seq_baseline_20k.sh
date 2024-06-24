mkdir -p logs_slurm/polybert
LLsub scripts/preprocess_image2seq.sh -s 8 -T 1:00:00 -o logs_slurm/polybert/preprocess
