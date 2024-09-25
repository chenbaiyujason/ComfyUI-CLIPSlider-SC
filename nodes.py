import torch
import random
from tqdm import tqdm
from comfy.model_base import BaseModel
from comfy.sd import load_checkpoint_guess_config
from server import PromptServer

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

class CLIPSliderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent_direction": ("LATENT",),
                "scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "prompt": ("STRING", {"default": "a photo of a person"}),
            },
            "optional": {
                "latent_direction_2nd": ("LATENT",),
                "scale_2nd": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_clip_slider"
    CATEGORY = "conditioning"

    def apply_clip_slider(self, model, clip, latent_direction, scale, prompt,
                          latent_direction_2nd=None, scale_2nd=0.0):
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        positive_cond = cond + latent_direction * scale
        if latent_direction_2nd is not None:
            positive_cond = positive_cond + latent_direction_2nd * scale_2nd

        negative_cond = cond - latent_direction * scale
        if latent_direction_2nd is not None:
            negative_cond = negative_cond - latent_direction_2nd * scale_2nd

        positive_conditioning = [[positive_cond, {"pooled_output": pooled}]]
        negative_conditioning = [[negative_cond, {"pooled_output": pooled}]]

        return (positive_conditioning, negative_conditioning)

class FindLatentDirectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "target_word": ("STRING", {"default": "happy"}),
                "opposite": ("STRING", {"default": "sad"}),
                "iterations": ("INT", {"default": 300, "min": 1, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "find_latent_direction"
    CATEGORY = "conditioning"

    def find_latent_direction(self, clip, target_word, opposite, iterations=300, seed=0):
        torch.manual_seed(seed)
        with torch.no_grad():
            positives = []
            negatives = []
            for _ in tqdm(range(iterations)):
                medium = random.choice(MEDIUMS)
                subject = random.choice(SUBJECTS)
                pos_prompt = f"a {medium} of a {target_word} {subject}"
                neg_prompt = f"a {medium} of a {opposite} {subject}"
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
        return (avg_diff,)

NODE_CLASS_MAPPINGS = {
    "CLIPSlider": CLIPSliderNode,
    "FindLatentDirection": FindLatentDirectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSlider": "CLIP Slider",
    "FindLatentDirection": "Find Latent Direction"
}
