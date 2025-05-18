### Evaluation 

> Cloned from [original s1 repo](https://github.com/simplescaling/s1/tree/main/eval) at commit `31a10f2481cb6708e4afa4154f9d74acd5dd70f8`.

We cloned [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at commit `4cec66e4e468d15789473d6d63c3a61a751fa524` and modified it. Setup:
```bash
cd eval/lm-evaluation-harness
pip install -e .[math,vllm]
```

All commands are in `eval/commands.sh`. For AIME24 we always pick the `aime24_nofigures` result, which uses a dataset that only contains the AIME24 figures if they are important for the task.

If you want to compute statistics (avg thinking tokens etc) for an evaluation run you can use 
`python eval/compute_sample_stats.py path_to_samples_file.jsonl`

All our evaluation result files are at: https://hf.co/datasets/simplescaling/results

To run REBASE: commands are in `eval/rebase/run.sh`
Note that for the evaluations in the Discussion section with REBASE we used https://huggingface.co/simplescaling/step-conditional-control-old trained on an older version of our dataset https://huggingface.co/datasets/simplescaling/s1K-step-conditional-control-old and run on an older version of our evaluation using https://huggingface.co/datasets/Maxwell-Jia/AIME_2024.