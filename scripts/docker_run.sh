export HUB_TOKEN=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
export WANDB_API_KEY=74e8368d36e7abf19bc9d56a66e74a7bea1033e5
if [[ -n "$1" ]]; then
    docker run -it --entrypoint /bin/bash --name $1 --gpus all -v .:/workspace parthpratim27/data-mixtures
else
    docker run -it --entrypoint /bin/bash --gpus all -v .:/workspace parthpratim27/data-mixtures
fi
