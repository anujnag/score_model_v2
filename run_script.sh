sbatch --partition=atlas \
        --gres=gpu:1 --mem=12G \
        --job-name="score_matching" --output="score_matching.out" \
        --wrap="python3 /atlas/u/aamdekar/score_matching++/score_matching_refined/main.py"