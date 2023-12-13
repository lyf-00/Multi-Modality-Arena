python eval_tiny.py \
    --model-name LLaVA-v1.5 \
    --device 0 \
    --batch-size 1 \
    --dataset-names CIFAR10,ImageNet,TextVQA,OKVQA,GQA,IconQA,VSR,ScienceQA,VizWiz,ImageNetVC_color,ImageNetVC_shape,ImageNetVC_material,ImageNetVC_component,ImageNetVC_others,VCR1_MCI,VCR1_OC \
    --sampled-root /home/v-liuyifei/Multimodal-Quantization/datasets/tiny_lvlm_datasets \
    --answer_path ./answer/llava-v1.5-7b \
    # --use-sampled 

python updated_eval_tiny.py \
    --model-name LLaVA \
    --device 1 \
    --batch-size 4 \
    --sampled-root /home/v-liuyifei/Multimodal-Quantization/datasets/tiny_lvlm_datasets \
    --answer-path ./answer/llava-v1.5-7b \


python eval_tiny.py \
    --model-name LLaVA-v1.5 \
    --device 0 \
    --batch-size 1 \
    --dataset-names TextVQA \
    --answer_path ./answer/llava-v1.5-7b \
    --use-sampled \
    --sample-seed 0 \
    --sample-num 5000 
    
python eval_tiny.py \
    --model-name LLaVA-v1.5 \
    --device 0 \
    --batch-size 1 \
    --dataset-names TextVQA \
    --answer_path ./answer/llava-v1.5-7b \
    --use-sampled \
    --sample-seed 0 \
    --sample-num 5000 



python eval_tiny.py \
    --model-name LLaVA-v1.5 \
    --device 1 \
    --batch-size 1 \
    --dataset-names TextVQA \
    --answer_path ./answer/llava-v1.5-7b-text \
    --use-sampled \
    --sample-seed 0 \
    --sample-num 5000 \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut.text",act_token_split=0

python eval_tiny.py \
    --model-name LLaVA-v1.5 \
    --device 2 \
    --batch-size 1 \
    --dataset-names TextVQA \
    --answer_path ./answer/llava-v1.5-7b-vision \
    --use-sampled \
    --sample-seed 0 \
    --sample-num 5000 \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut.vision",act_token_split=0

python eval_tiny.py \
    --model-name LLaVA-v1.5 \
    --device 3 \
    --batch-size 1 \
    --dataset-names TextVQA \
    --answer_path ./answer/llava-v1.5-7b-vision \
    --use-sampled \
    --sample-seed 0 \
    --sample-num 5000 \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut",act_token_split=1


python eval_tiny.py \
    --model-name LLaVA-v1.5 \
    --device 4 \
    --batch-size 1 \
    --dataset-names TextVQA \
    --answer_path ./answer/llava-v1.5-7b-hybrid \
    --use-sampled \
    --sample-seed 0 \
    --sample-num 5000 \
    --quant_args w_bits=16,a_bits=4,act_quant_func="lut.hybrid",act_token_split=0

