import sys
sys.path.append('/home/tianze/MLLM/HuatuoGPT-Vision')

from cli import HuatuoChatbot
import torch
import json
from tqdm import tqdm
import os
import numpy as np
from utils import attention_ratio_vectorized, kl_divergence_vectorized, js_divergence_vectorized
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["7B", "34B"], required=True, help="Model size (7B or 34B).")
    parser.add_argument("--dataset", type=str, choices=["VGMED", "COCO"], required=True)
    parser.add_argument("--q_type", type=str, choices=["localization", "attribute"], required=True)
    return parser.parse_args()

args = parse_args()
model_size = args.model
dataset = args.dataset
q_type = args.q_type

bot = HuatuoChatbot(f"FreedomIntelligence/HuatuoGPT-Vision-{model_size}")
LAYERS = bot.model.config.num_hidden_layers
NUM_IMG_TOKENS = 576
PATCHES = 24
SIZE = (336, 336)


def generate_attention_maps(question, image_path, layers=range(LAYERS)):

    general_question = 'Write a general description of the image.'
    prompt = f"{question} Answer the question using a single word or phrase."
    general_prompt = f"{general_question} Answer the question using a single word or phrase."

    model_output, input_ids = bot.inference_with_attention_output(prompt,image_path)
    input_ids = input_ids[0].cpu()
    index = torch.where(input_ids==-200)[0]
    att_maps = np.array([model_output['attentions'][layer][0, :, -1, index:index+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy() for layer in layers])

    model_output, input_ids = bot.inference_with_attention_output(general_prompt,image_path)
    input_ids = input_ids[0].cpu()
    index = torch.where(input_ids==-200)[0]
    general_att_maps = np.array([model_output['attentions'][layer][0, :, -1, index:index+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy() for layer in layers])

    return att_maps, general_att_maps


attention_ratios_normalized = []
attention_kl_normalized = []
attention_js_normalized = []

questions = []
input_path = f"./{dataset}_{q_type}.jsonl"
data_path = "/home/tianze/MLLM/HuatuoGPT-Vision/val2014" if dataset == "COCO" else "/home/tianze/Datasets/SAMed2D_v1/selected"
with open(input_path, "r") as infile:
    for line in infile:
        questions.append(json.loads(line))


for sample in tqdm(questions):
    image_path = os.path.join(data_path, sample["image"])
    qs = sample["question"]
    gt_tokens = sample["bbox_gt_tokens"]
    att_maps, general_att_maps = generate_attention_maps(qs, image_path)

    att_ratio_normalized = attention_ratio_vectorized(att_maps/general_att_maps, gt_tokens)
    att_kl_normalized = kl_divergence_vectorized(att_maps/general_att_maps, gt_tokens)
    att_js_normalized = js_divergence_vectorized(att_maps/general_att_maps, gt_tokens)

    attention_ratios_normalized.append(att_ratio_normalized)
    attention_kl_normalized.append(att_kl_normalized)
    attention_js_normalized.append(att_js_normalized)


os.makedirs(f"./results/{model_size}/layer_{dataset}/{q_type}", exist_ok=True)
np.save(f"./results/{model_size}/layer_{dataset}/{q_type}/attention_ratios.npy", np.array(attention_ratios_normalized))
np.save(f"./results/{model_size}/layer_{dataset}/{q_type}/attention_kl.npy", np.array(attention_kl_normalized))
np.save(f"./results/{model_size}/layer_{dataset}/{q_type}/attention_js.npy", np.array(attention_js_normalized))