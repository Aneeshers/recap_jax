gpu_list=$1
WANDB_API_KEY=$(cat ./dev/wandb_key)

if [ -z "$gpu_list" ]; then
    gpu_list="all"
fi

gpu_flag="--runtime=nvidia"

docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu_list \
    -e NVIDIA_VISIBLE_DEVICES=$gpu_list \
    $gpu_flag \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/recap_jax \
    --name recap_jax\_${gpu_list//,/} \
    --user $(id -u) \
    -itd recap_jax bash
