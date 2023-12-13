python eval_tiny.py \
    --model-name LLaVA \
    --device 0 \
    --batch-size 4 \
    --dataset-names TextVQA \
    --sampled-root /home/v-liuyifei/Multimodal-Quantization/datasets/tiny_lvlm_datasets \
    --answer_path ./answer/llava-v1.5-7b \
    # --use-sampled 

python updated_eval_tiny.py \
    --model-name LLaVA \
    --device 1 \
    --batch-size 4 \
    --sampled-root /home/v-liuyifei/Multimodal-Quantization/datasets/tiny_lvlm_datasets \
    --answer-path ./answer/llava-v1.5-7b \
