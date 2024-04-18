
run-full-transfer-vanillatransformer:
	python3 run_benchmark_transfer_learning_fulldata.py --model VanillaTransformer --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-informer:
	python3 run_benchmark_transfer_learning_fulldata.py --model Informer           --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-autoformer:
	python3 run_benchmark_transfer_learning_fulldata.py --model Autoformer         --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-fedformer:
	python3 run_benchmark_transfer_learning_fulldata.py --model FEDformer          --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-patchtst:
	python3 run_benchmark_transfer_learning_fulldata.py --model PatchTST           --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

#
# Small models
#
run-full-transfer-small-vanillatransformer:
	python3 run_benchmark_transfer_learning_fulldata.py --model VanillaTransformer --small --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-small-informer:
	python3 run_benchmark_transfer_learning_fulldata.py --model Informer           --small --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-small-autoformer:
	python3 run_benchmark_transfer_learning_fulldata.py --model Autoformer         --small --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-small-fedformer:
	python3 run_benchmark_transfer_learning_fulldata.py --model FEDformer          --small --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2

run-full-transfer-small-patchtst:
	python3 run_benchmark_transfer_learning_fulldata.py --model PatchTST           --small --batch 256 --epochs 10000 --lr 0.0001 --stop 4 --valsteps 1024  --iterations 2



#
# Single serie models
#
run-singleserie-vanillatransformer:
	python3 run_benchmark_transformers.py --model VanillaTransformer --epochs 100 --inputfactor 4 --lr 0.001 --batch 1024 --freq Weekly --procs 4