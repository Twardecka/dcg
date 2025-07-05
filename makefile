all: 
	clear 
	# CUDA_VISIBLE_DEVICES=0 python src/main.py --config=dcg --env-config=sc2 with env_args.map_name=3s5z seed=100 use_cuda=True 
	CUDA_VISIBLE_DEVICES=0 python src/main.py --config=dcg --env-config=gather with seed=100 use_cuda=False 

run_baselines: 
	for f in runs_baselines/*.job; do sbatch $$f; done 