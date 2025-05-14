# Data-Mixtures : Official Repository for Paper Towards Neurips/ICLR/EMNLP/...

## Environment Setup (Docker)
```bash
bash scripts/docker_build.sh
```

To start a session with the built docker, run the following command. This mounts the entire code directory inside the docker.

```bash
bash scripts/docker_run.sh
```


## Directory Structure & Naming Convention

The struture of the root should be the following
```bash
|- checkpoints
    |- full_ft
        |- ...
        |- result_gpt2_<subtask-name>-<json-file-name-without-dot-json-extension>
        |- ...

|- data
    |- <subtask-name>
        |-<subtask-name>-<json-file-name-without-dot-json-extension>.json
    |- to
        |- ...
    |-flan2021
        |- ...
```


## Run a Finetuning Experiment
### T0 Fine-Tuning over GPT2 (Use this as a base to build scripts for other tasks as well)
```bash
bash scripts/finetuning/t0_all_tasks.sh
```

### Llama Fine-Tuning over created mixture
```bash
bash scripts/finetuning/llama_ft.sh
```

## Creating Final Data-Mixtures

### Uniform Data-Mixtures
```bash
python -m src.create_uniform_mix
```

## PMI Similarity Matrix Calculation (Multi-GPU)
```bash
python -m src.metrics.pmi
```

## JSD Similarity Artifacts
```bash
https://1drv.ms/f/c/0c180139dc7c7daa/EhBiTsX77C1DrFJ8PhFbuooB-p5Y8Wf1lt50fsyB8cIrvA?e=sFUpts
```

