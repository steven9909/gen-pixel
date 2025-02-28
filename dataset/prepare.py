import argparse
import cv2
from skimage.segmentation import quickshift
import numpy as np
import json
import lmdb
from functools import reduce
import pickle
from tqdm import tqdm
import os
import glob


def save_lmdb(save_dir, associations):
    env = lmdb.open(save_dir, map_size=1e9)

    with env.begin(write=True) as txn:
        for idx, (image, text) in enumerate(tqdm(associations)):
            if image is None:
                print(f"Warning: Unable to read {image_path}, skipping.")
                continue

            _, image_bytes = cv2.imencode(".png", image)

            data = {"image": image_bytes.tobytes(), "text": text}
            txn.put(str(idx).encode(), pickle.dumps(data))

    env.close()
    print(f"LMDB dataset created at {save_dir}")


def apply_average_color(image, segmentation):
    output = image.copy()

    region_labels = np.unique(segmentation)

    for label in region_labels:
        mask = segmentation == label

        avg_color = image[mask].mean(axis=0)

        output[mask] = avg_color.astype(image.dtype)

    return output


def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segments_quick = quickshift(image, kernel_size=2, max_dist=6, ratio=0.5)

    return apply_average_color(image, segments_quick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", type=str, required=True, nargs="+")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save_lmdb", action="store_true")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--discard_outliers", action="store_true")
    args = parser.parse_args()

    image_paths = [
        glob.glob(os.path.join(image_dir, "*.jpg")) for image_dir in args.image_dirs
    ]

    total_association = []

    with tqdm(total=reduce(lambda c, l: c + len(l), image_paths, 0)) as pbar:
        counter = 0
        for i, image_dir in enumerate(args.image_dirs):

            association = json.load(open(f"{image_dir}/association.json"))

            for image_path in image_paths[i]:
                processed_image = process_image(image_path)

                image_name = os.path.basename(image_path)

                if args.save_lmdb:
                    total_association.append((processed_image, association[image_name]))
                else:
                    cv2.imwrite(f"{args.save_dir}/{counter}.png", processed_image)
                    total_association.append(
                        (f"{counter}.png", association[image_name])
                    )

                counter += 1

                pbar.update(1)

    if args.save_lmdb:
        save_lmdb(os.path.join(args.save_dir, f"{args.name}.lmdb"), total_association)
    else:
        pickle.dump(total_association, open(f"{args.save_dir}/association.pkl", "wb"))
