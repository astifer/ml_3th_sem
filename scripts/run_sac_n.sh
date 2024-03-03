export WANDB_ENTITY="astifer"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="b89cebf2471bbc17859484af40db8b488c41af70"

cd /workspace/ml_3th_sem/algorithms

python3 edac.py \
  --eta=0.0 \
  --actor_learning_rate=0.0003 \
  --alpha_learning_rate=0.0003 \
  --batch_size=256 \
  --buffer_size=2000000 \
  --critic_learning_rate=0.0003 \
  --device="cuda" \
  --env_name="halfcheetah-medium-expert-v2" \
  --eval_episodes=10 \
  --eval_every=5 \
  --eval_seed=42 \
  --gamma=0.99 \
  --group="sac-n-halfcheetah-medium-expert-v2-multiseed-v2" \
  --hidden_dim=256 \
  --log_every=100 \
  --max_action=1.0 \
  --name="SAC-N" \
  --num_critics=10 \
  --num_epochs=3000 \
  --num_updates_on_epoch=1000 \
  --project="offline_rl" \
  --tau=0.005 \
  --train_seed=10