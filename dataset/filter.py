import torch
from PIL import Image
import clip
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    association = json.load(open(f"{args.image_dir}/association.json"))

    image_dict = defaultdict(list)

    for k, v in association.items():
        image_dict[v].append(k)

    model, preprocess = clip.load("ViT-B/32", device=args.device)

    with torch.no_grad():
        for k in image_dict.keys():
            image_names = image_dict[k]

            text_token = clip.tokenize(k).to(args.device)
            text_feature = model.encode_text(text_token)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

            sims = []

            for image_name in image_names:
                path = f"{args.image_dir}{os.path.sep}{image_name}"

                image = Image.open(path)
                image_feature = preprocess(image).unsqueeze(0).to(args.device)

                image_feature = model.encode_image(image_feature)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)

                sims.append(
                    (image_name, (image_feature @ text_feature.T).squeeze().item())
                )

            std = np.array([sim for _, sim in sims]).std()
            mean = np.array([sim for _, sim in sims]).mean()

            for image_name, sim in sims:
                if sim <= mean - 2 * std or sim < 0.25:
                    print(f"Outlier: {image_name} with similarity {sim}, name: {k}")
