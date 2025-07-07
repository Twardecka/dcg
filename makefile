all:
	 clear
	 CUDA_VISIBLE_DEVICES=1 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=1 use_cuda=True &
	 CUDA_VISIBLE_DEVICES=1 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=2 use_cuda=True 
	 CUDA_VISIBLE_DEVICES=2 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=3 use_cuda=True &
	 CUDA_VISIBLE_DEVICES=2 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=4 use_cuda=True 
	 CUDA_VISIBLE_DEVICES=3 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=5 use_cuda=True &
	 CUDA_VISIBLE_DEVICES=3 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=6 use_cuda=True 
	 CUDA_VISIBLE_DEVICES=4 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=7 use_cuda=True &
	 CUDA_VISIBLE_DEVICES=4 python src/main.py --config=dcg --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" seed=8 use_cuda=True 

run_baselines: 
	for f in runs_baselines/*.job; do sbatch $$f; done 

