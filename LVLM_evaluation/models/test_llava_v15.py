import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel, StoppingCriteria
# from .llava import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from . import get_image



DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def get_model_name(model_path):
    # get model name
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
    
    return model_name


def get_conv(model_name):
    if "llava" in model_name.lower():
        if "v1" in model_name.lower():
            template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt_multimodal"
        else:
            template_name = "multimodal"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "koala" in model_name: # Hardcode the condition
        template_name = "bair_v1"
    elif "v1" in model_name:    # vicuna v1_1/v1_2
        template_name = "vicuna_v1_1"
    else:
        template_name = "v1"
    return conv_templates[template_name].copy()


class TestLLaVA:
    def __init__(self, device=None):
        # model_path="liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        model_path="liuhaotian/llava-v1.5-7b"
        model_name = get_model_name_from_path(model_path)
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_model(model_path, model_name)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name) # model_base is None
        # self.conv = get_conv(model_name)
        self.conv = conv_templates['vicuna_v1'].copy()
        self.image_process_mode = "Resize" # Crop, Resize, Pad

        if device is not None:
            self.move_to_device(device)
        
    def move_to_device(self, device=None):
        # if device is not None and 'cuda' in device.type:
        #     self.dtype = torch.float16
        #     self.device = device
        # else:
        #     self.dtype = torch.float32
        #     self.device = 'cpu'
        # vision_tower = self.model.get_model().vision_tower[0]
        # vision_tower.to(device=self.device, dtype=self.dtype)
        # self.model.to(device=self.device, dtype=self.dtype)
        pass
    # @torch.no_grad()
    # def generate(self, image, question, max_new_tokens=256):
    #     image = get_image(image)
    #     conv = self.conv.copy()
    #     text = question + '\n<image>'
    #     text = (text, image, self.image_process_mode)
    #     conv.append_message(conv.roles[0], text)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    #     stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
    #     output = self.do_generate([prompt], [image], stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)[0]

    #     return output

    @torch.no_grad()
    def pure_generate(self, image, question, max_new_tokens=256):
        image = get_image(image)
        conv = self.conv.copy()
        prompt = question
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        output = self.do_generate([prompt], [image], stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)[0]

        return output
    
    # @torch.no_grad()
    # def batch_generate(self, image_list, question_list, max_new_tokens=256):
    #     images, prompts = [], []
    #     for image, question in zip(image_list, question_list):
    #         image = get_image(image)
    #         conv = self.conv.copy()
    #         text = question + '\n<image>'
    #         text = (text, image, self.image_process_mode)
    #         conv.append_message(conv.roles[0], text)
    #         conv.append_message(conv.roles[1], None)
    #         prompt = conv.get_prompt()
    #         prompts.append(prompt)
    #         images.append(image)
    #     stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
    #     outputs = self.do_generate(prompts, images, stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)

    #     return outputs
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        images, prompts = [], []
        for image, question in zip(image_list, question_list):
            image = get_image(image)
            # line = self.questions[index]
            # image_file = line["image"]
            # qs = line["text"]
            qs = question
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = self.conv.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            self.model.img_token_start = list(input_ids).index(IMAGE_TOKEN_INDEX)
            
            # return input_ids, image_tensor
            # get input_ids, image_tensor
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            input_ids = input_ids.unsqueeze(0)
            image_tensor = image_tensor.unsqueeze(0)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    # do_sample=True if args.temperature > 0 else False,
                    # temperature=args.temperature,
                    # top_p=args.top_p,
                    # num_beams=args.num_beams,
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True)
            # conv = self.conv.copy()
            # text = question + '\n<image>'
            # text = (text, image, self.image_process_mode)
            # conv.append_message(conv.roles[0], text)
            # conv.append_message(conv.roles[1], None)
            # prompt = conv.get_prompt()
            # prompts.append(prompt)
            # images.append(image)
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        # # outputs = self.do_generate(prompts, images, stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)
        # outputs = self.do_generate(prompts, images, stop_str=stop_str, max_new_tokens=max_new_tokens)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        return [outputs]

    # @torch.no_grad()
    # def do_generate(self, prompts, images, dtype=torch.float16, temperature=0.2, max_new_tokens=256, stop_str=None, keep_aspect_ratio=False):
    #     if keep_aspect_ratio:
    #         new_images = []
    #         for image, prompt in zip(images, prompts):
    #             max_hw, min_hw = max(image.size), min(image.size)
    #             aspect_ratio = max_hw / min_hw
    #             max_len, min_len = 448, 224
    #             shortest_edge = int(min(max_len / aspect_ratio, min_len))
    #             image = self.image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
    #             new_images.append(image.to(self.model.device, dtype=dtype))
    #             # replace the image token with the image patch token in the prompt (each occurrence)
    #             cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
    #             replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
    #             if getattr(self.model.config, 'mm_use_im_start_end', False):
    #                 replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    #             prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
    #         images = new_images
    #     else:
    #         images = self.image_processor(images, return_tensors='pt')['pixel_values']
    #         images = images.to(self.model.device, dtype=dtype)
    #         replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
    #         if getattr(self.model.config, 'mm_use_im_start_end', False):
    #             replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    #         prompts = [prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token) for prompt in prompts]

    #     stop_idx = None
    #     if stop_str is not None:
    #         stop_idx = self.tokenizer(stop_str).input_ids
    #         if len(stop_idx) == 1:
    #             stop_idx = stop_idx[0]
    #         else:
    #             stop_idx = None

    #     input_ids = self.tokenizer(prompts).input_ids
    #     batch_size = len(input_ids)
    #     min_prompt_size = min([len(input_id) for input_id in input_ids])
    #     max_prompt_size = max([len(input_id) for input_id in input_ids])
    #     for i in range(len(input_ids)):
    #         padding_size = max_prompt_size - len(input_ids[i])
    #         # input_ids[i].extend([self.tokenizer.pad_token_id] * padding_size)
    #         input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]
        
    #     output_ids = []
    #     get_result = [False for _ in range(batch_size)]
    #     for i in range(max_new_tokens):
    #         if i == 0:
    #             out = self.model(
    #                 torch.as_tensor(input_ids).to(self.model.device),
    #                 use_cache=True,
    #                 images=images)
    #             logits = out.logits
    #             past_key_values = out.past_key_values
    #         else:
    #             out = self.model(input_ids=token,
    #                         use_cache=True,
    #                         attention_mask=torch.ones(batch_size, past_key_values[0][0].shape[-2] + 1, device=self.model.device),
    #                         past_key_values=past_key_values)
    #             logits = out.logits
    #             past_key_values = out.past_key_values

    #         last_token_logits = logits[:, -1]
    #         if temperature < 1e-4:
    #             token = torch.argmax(last_token_logits, dim=-1)
    #         else:
    #             probs = torch.softmax(last_token_logits / temperature, dim=-1)
    #             token = torch.multinomial(probs, num_samples=1)
    #         token = token.long().to(self.model.device)

    #         output_ids.append(token)
    #         for idx in range(len(token)):
    #             if token[idx] == stop_idx or token[idx] == self.tokenizer.eos_token_id:
    #                 get_result[idx] = True
    #         if all(get_result):
    #             break
        
    #     output_ids = torch.cat(output_ids, dim=1).long()
    #     outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #     if stop_str is not None:
    #         for i in range(len(outputs)):
    #             pos = outputs[i].rfind(stop_str)
    #             if pos != -1:
    #                 outputs[i] = outputs[i][:pos]
        
    #     return outputs