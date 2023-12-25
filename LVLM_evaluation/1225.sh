python eval.py \
    --model_name LLaVA-v1.5 \
    --device 0 \
    --batch_size 1 \
    --dataset_name CIFAR10 \
    --question "Classify the main object in the image." \
    --max_new_tokens 64 \
    --answer_path 'answer/llava-v1.5-7b' \
    --eval_cls \
    # --sample_num 20
        # --question "The photo of the" \
# please check the name of models/datasets in (models/task_datasets)/__init__.py
# do not need to specific question and max_new_tokens in default

python eval.py \
    --model_name LLaVA-v1.5 \
    --device 1 \
    --batch_size 1 \
    --dataset_name CIFAR10 \
    --question "Classify the main object in the image." \
    --max_new_tokens 64 \
    --answer_path 'answer/llava-v1.5-7b-text-new' \
    --eval_cls \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut.text",act_token_split=0


python eval.py \
    --model_name LLaVA-v1.5 \
    --device 2 \
    --batch_size 1 \
    --dataset_name CIFAR10 \
    --question "Classify the main object in the image." \
    --max_new_tokens 64 \
    --answer_path 'answer/llava-v1.5-7b-vision-new' \
    --eval_cls \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut.vision",act_token_split=0

python eval.py \
    --model_name LLaVA-v1.5 \
    --device 3 \
    --batch_size 1 \
    --dataset_name CIFAR10 \
    --question "Classify the main object in the image." \
    --max_new_tokens 64 \
    --answer_path 'answer/llava-v1.5-7b-split-new' \
    --eval_cls \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut",act_token_split=1

python eval.py \
    --model_name LLaVA-v1.5 \
    --device 4 \
    --batch_size 1 \
    --dataset_name CIFAR10 \
    --question "Classify the main object in the image." \
    --max_new_tokens 64 \
    --answer_path 'answer/llava-v1.5-7b-hybrid-new' \
    --eval_cls \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut.hybrid",act_token_split=0
