# M4-Competition-TimeTransformers

pip3 install -r requirements.txt


## Transfer
- VanillaTransformer
- Informer
- Autoformer
- FEDformer
- PatchTST


## Runs

- exp #1
    python3 run_benchmark_transfer_learning.py --freq Weekly --model VanillaTransformer --epochs 2000 --inputfactor 4
- exp #2
    python3 run_benchmark_transfer_learning.py --freq Weekly --model VanillaTransformer --epochs 2000 --inputfactor 8



python3 run_benchmark_transfer_learning.py --freq Weekly --model VanillaTransformer --epochs 2000 --inputfactor 4 --small
