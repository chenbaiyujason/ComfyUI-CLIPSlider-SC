import torch
import random
from tqdm import tqdm
from comfy.model_base import BaseModel
from comfy.sd import load_checkpoint_guess_config
from server import PromptServer

# 常量定义
MEDIUMS = [
    "painting", "drawing", "photograph", "HD photo", "illustration", "portrait",
    "sketch", "3d render", "digital painting", "concept art", "screenshot",
    "canvas painting", "watercolor art", "print", "mosaic", "sculpture",
    "cartoon", "comic art", "anime",
]

SUBJECTS = [
    "dog", "cat", "horse", "cow", "pig", "sheep", "lion", "elephant", "monkey",
    "bird", "chicken", "eagle", "parrot", "penguin", "fish", "shark", "dolphin",
    "whale", "octopus", "bee", "butterfly", "ant", "ladybug", "person", "man",
    "woman", "child", "baby", "boy", "girl", "car", "boat", "airplane", "bicycle",
    "motorcycle", "train", "building", "house", "bridge", "castle", "temple",
    "monument", "tree", "flower", "mountain", "lake", "river", "ocean", "beach",
    "fruit", "vegetable", "meat", "bread", "cake", "soup", "coffee", "toy", "book",
    "phone", "computer", "TV", "camera", "musical instrument", "furniture", "road",
    "park", "garden", "forest", "city", "sunset", "clouds",
]

# 辅助类
class PromptFormatter:
    def __init__(self, format_string):
        self.format_string = format_string
    
    def format(self, **kwargs):
        return self.format_string.format(**kwargs)

# 输入处理节点
class SliderObjectIn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mediums": ("STRING", {"multiline": True}),
                "subjects": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("MEDIUMS", "SUBJECTS")
    FUNCTION = "process_input"
    CATEGORY = "conditioning"

    def process_input(self, mediums, subjects):
        def split_string(s):
            if ";" in s:
                return [item.strip() for item in s.split(";") if item.strip()]
            elif s.strip():
                return [s.strip()]
            else:
                return []
        
        return (split_string(mediums), split_string(subjects))

# 提示格式化节点
class PromptFormatterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pos_format_string": ("STRING", {"default": "a {medium} of a {word} {subject}", "multiline": True}),
                "neg_format_string": ("STRING", {"default": "a {medium} of a {word} {subject}", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("PROMPT_FORMATTER", "PROMPT_FORMATTER")
    RETURN_NAMES = ("POSITIVE_PROMPT_FORMATTER", "NEGATIVE_PROMPT_FORMATTER")
    FUNCTION = "create_formatter"
    CATEGORY = "conditioning"

    def create_formatter(self, pos_format_string, neg_format_string):
        return (PromptFormatter(pos_format_string), PromptFormatter(neg_format_string))

# 潜在方向查找节点
class FindLatentDirectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "target_word": ("STRING", {"default": "happy"}),
                "opposite_word": ("STRING", {"default": "sad"}),
                "iterations": ("INT", {"default": 30, "min": 1, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "positive_formatter": ("PROMPT_FORMATTER",),
                "negative_formatter": ("PROMPT_FORMATTER",),
                "mediums": ("MEDIUMS",),
                "subjects": ("SUBJECTS",),
            }
        }

    RETURN_TYPES = ("LATENT","FLOAT")
    RETURN_NAMES = ("latent","word_distance")
    FUNCTION = "find_latent_direction"
    CATEGORY = "conditioning"

    def find_latent_direction(self, clip, target_word, opposite_word, iterations=300, seed=0, positive_formatter=None, negative_formatter=None, mediums=None, subjects=None):
        torch.manual_seed(seed)
        with torch.no_grad():
            positives = []
            negatives = []
            
            # 如果没有提供 mediums 或 subjects,使用默认值
            if not mediums:
                mediums = MEDIUMS
            if not subjects:
                subjects = SUBJECTS
            
            for _ in tqdm(range(iterations)):
                medium = random.choice(mediums)
                subject = random.choice(subjects)
                if positive_formatter:
                    pos_prompt = positive_formatter.format(medium=medium, word=target_word, subject=subject)
                else:
                    pos_prompt = f"a {medium} of a {target_word} {subject}"
                
                if negative_formatter:
                    neg_prompt = negative_formatter.format(medium=medium, word=opposite_word, subject=subject)
                else:
                    neg_prompt = f"a {medium} of a {opposite_word} {subject}"
                pos_toks = clip.tokenize(pos_prompt)
                neg_toks = clip.tokenize(neg_prompt)
                pos = clip.encode_from_tokens(pos_toks)
                neg = clip.encode_from_tokens(neg_toks)
                positives.append(pos)
                negatives.append(neg)

        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        diffs = positives - negatives
        avg_diff = diffs.mean(0, keepdim=True)
        distance = torch.norm(avg_diff).item()
        return (avg_diff,distance)


# 潜在方向查找节点
class FindLatentDirectionNodePooled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "target_word": ("STRING", {"default": "happy"}),
                "opposite_word": ("STRING", {"default": "sad"}),
                "iterations": ("INT", {"default": 30, "min": 1, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "positive_formatter": ("PROMPT_FORMATTER",),
                "negative_formatter": ("PROMPT_FORMATTER",),
                "mediums": ("MEDIUMS",),
                "subjects": ("SUBJECTS",),
            }
        }

    RETURN_TYPES = ("LATENT","FLOAT")
    RETURN_NAMES = ("latent","word_distance")
    FUNCTION = "find_latent_direction"
    CATEGORY = "conditioning"

    def find_latent_direction(self, clip, target_word, opposite_word, iterations=300, seed=0, positive_formatter=None, negative_formatter=None, mediums=None, subjects=None):
        torch.manual_seed(seed)
        with torch.no_grad():
            positives = []
            negatives = []
            
            # 如果没有提供 mediums 或 subjects,使用默认值
            if not mediums:
                mediums = MEDIUMS
            if not subjects:
                subjects = SUBJECTS
            
            for _ in tqdm(range(iterations)):
                medium = random.choice(mediums)
                subject = random.choice(subjects)
                if positive_formatter:
                    pos_prompt = positive_formatter.format(medium=medium, word=target_word, subject=subject)
                else:
                    pos_prompt = f"a {medium} of a {target_word} {subject}"
                
                if negative_formatter:
                    neg_prompt = negative_formatter.format(medium=medium, word=opposite_word, subject=subject)
                else:
                    neg_prompt = f"a {medium} of a {opposite_word} {subject}"
                pos_toks = clip.tokenize(pos_prompt)
                neg_toks = clip.tokenize(neg_prompt)
                #flux的返回是t5_out(no t5_pooled), l_pooled(no l_out)
                pos,pos_pooled = clip.encode_from_tokens(pos_toks,return_pooled=True)
                neg,neg_pooled = clip.encode_from_tokens(neg_toks,return_pooled=True)
                positives.append(pos_pooled)
                negatives.append(neg_pooled)

        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)
        diffs = positives - negatives
        avg_diff = diffs.mean(0, keepdim=True)
        distance = torch.norm(avg_diff).item()
        return (avg_diff,distance)



# CLIP滑块应用节点
class CLIPSliderNodePooled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "latent_direction": ("LATENT",),
                "slider_target": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "prompt": ("STRING", {"default": "a photo of a person", "multiline": True}),
                "guidance": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "latent_direction_2nd": ("LATENT",),
                "slider_target_2nd": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_clip_slider"
    CATEGORY = "conditioning"

    def apply_clip_slider(self,  clip, latent_direction, slider_target, prompt,guidance,
                          latent_direction_2nd=None, slider_target_2nd=0.0):
        tokens = clip.tokenize(prompt)
        #flux的返回是(t5_out(no t5_pooled), l_pooled(no l_out))
        positive_conditioning = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        negative_conditioning = positive_conditioning.copy()
        negative_conditioning['pooled_output'] = positive_conditioning['pooled_output'] - latent_direction * slider_target

        positive_conditioning['pooled_output'] = positive_conditioning['pooled_output'] + latent_direction * slider_target
        if latent_direction_2nd is not None:
            positive_conditioning['pooled_output'] = positive_conditioning['pooled_output'] + latent_direction_2nd * slider_target_2nd

        if latent_direction_2nd is not None:
            negative_conditioning['pooled_output'] = negative_conditioning['pooled_output'] - latent_direction_2nd * slider_target_2nd
        cond = positive_conditioning.pop("cond")
        positive_conditioning["guidance"] = guidance
        positive_output = ([cond, positive_conditioning], )
        negative_output = ([cond, negative_conditioning], )

        return (positive_output, negative_output)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SliderObjectIn": SliderObjectIn,
    "SliderPrompt": PromptFormatterNode,
    "SliderLatent": FindLatentDirectionNode,
    "SliderLatentPooled": FindLatentDirectionNodePooled,
    "CLIPSliderApply": CLIPSliderNode,
    "CLIPSliderApplyPooled": CLIPSliderNodePooled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SliderObjectIn": "SliderObjectIn",
    "SliderPrompt": "SliderPrompt",
    "SliderLatent": "SliderLatent",
    "SliderLatentPooled": "SliderLatentPooled",
    "CLIPSliderApply": "CLIPSliderApply",
    "CLIPSliderApplyPooled": "CLIPSliderApplyPooled",
}
