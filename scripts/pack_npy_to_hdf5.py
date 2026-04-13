"""Pack pickle + many small .npy files into a single HDF5 file.

Usage:
    python scripts/pack_npy_to_hdf5.py \
        --input ~/data/sft_data/meta/simplerenv_bridge_trainval.pkl \
        --output ~/data/sft_data/meta/simplerenv_bridge_trainval.h5
"""

import argparse
import pickle
import numpy as np
import h5py
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original pickle file")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    args = parser.parse_args()

    print(f"Loading pickle: {args.input}")
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    n_episodes = len(data)
    total_frames = sum(len(ep["image"]) for ep in data)
    print(f"Episodes: {n_episodes}, Total frames: {total_frames}")

    # Probe shape/dtype from first npy file
    sample = np.load(data[0]["image"][0])
    # Each npy is (1, 32, 32) — squeeze the leading dim
    frame_shape = sample.squeeze(0).shape  # (32, 32)
    dtype = sample.dtype
    print(f"Frame shape: {frame_shape}, dtype: {dtype}")

    # Probe action shape
    action_dim = data[0]["action"].shape[-1]
    print(f"Action dim: {action_dim}")

    with h5py.File(args.output, "w") as f:
        images_ds = f.create_dataset(
            "images", shape=(total_frames, *frame_shape), dtype=dtype
        )
        grippers_ds = f.create_dataset(
            "grippers", shape=(total_frames, *frame_shape), dtype=dtype
        )
        actions_ds = f.create_dataset(
            "actions", shape=(total_frames, action_dim), dtype=np.float64
        )
        offsets_ds = f.create_dataset(
            "episode_offsets", shape=(n_episodes, 2), dtype=np.int64
        )
        # Variable-length strings
        str_dt = h5py.string_dtype()
        texts_ds = f.create_dataset("texts", shape=(n_episodes,), dtype=str_dt)

        offset = 0
        for i, ep in enumerate(tqdm(data, desc="Packing episodes")):
            n_frames = len(ep["image"])

            # Load and write image frames
            for j, path in enumerate(ep["image"]):
                arr = np.load(path).squeeze(0)  # (32, 32)
                images_ds[offset + j] = arr

            # Load and write gripper frames
            for j, path in enumerate(ep["gripper_image"]):
                arr = np.load(path).squeeze(0)  # (32, 32)
                grippers_ds[offset + j] = arr

            # Write actions
            actions_ds[offset : offset + n_frames] = ep["action"][:n_frames]

            # Write metadata
            offsets_ds[i] = [offset, n_frames]
            texts_ds[i] = ep["text"]

            offset += n_frames

    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
