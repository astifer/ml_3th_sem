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


## Reproducing results
If you'd like to reproduce our results, take the steps analogous to ones we've covered below on the example of edac
```bash
