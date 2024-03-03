# Offline RL methods implementation with docker and wandb support


## Getting started
```bash
git clone https://github.com/astifer/ml_3th_sem.git && cd offline_rl
pip install -r requirements.txt

# alternatively, you could use docker
docker build -t <image_name> .
docker run --gpus=all -it --rm --name <container_name> <image_name>
```

## Implementations
| Algorithm                                                                                                                      | Variants Implemented                           |
|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| ✅ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(EDAC)](https://arxiv.org/abs/2110.01548) | [`offline/edac.py`](algorithms/edac.py)        |
| ✅ [Offline Reinforcement Learning with Implicit Q-Learning <br>(IQL)](https://arxiv.org/abs/2110.06169)                       | [`offline/iql.py`](algorithms/iql.py)          |
| ✅ [Accelerating Online Reinforcement Learning with Offline Datasets <br>(AWAC)](https://arxiv.org/abs/2006.09359)             | [`offline/awac.py`](algorithms/awac.py)        |
| ✅ [A Minimalist Approach to Offline Reinforcement Learning <br>(TD3+BC)](https://arxiv.org/abs/2106.06860)                    | [`offline/td3_bc.py`](algorithms/td3_bc.py)    |


## Reproducing results
If you'd like to run methods on your own, take the steps analogous to ones we've covered below on the example of bash script for edac
```bash
cd /workspace/ml_3th_sem/algorithms

python3 edac.py \
  --actor_learning_rate=0.0003 \
  --alpha_learning_rate=0.0003 \
  --batch_size=256 \
  --buffer_size=2000000 \
  --critic_learning_rate=0.0003 \
  --device=cuda \
  --env_name="halfcheetah-medium-expert-v2" \
  --eta=5.0 \
  --eval_episodes=10 \
  --eval_every=5 \
  --eval_seed=42 \
  --gamma=0.99 \
  --group="edac-halfcheetah-medium-expert-v2-multiseed-v2" \
  --hidden_dim=256 \
  --log_every=100 \
  --max_action=1.0 \
  --name="EDAC" \
  --num_critics=10 \
  --num_epochs=3000 \
  --num_updates_on_epoch=1000 \
  --project="offline_rl" \
  --tau=0.005 \
  --train_seed=10 \
```
Our configurations are available in the `configs` folder, The learning and evaluation processes are available via [wandb logs](https://wandb.ai/astifer/offline_rl?nw=nwuserastifer)
