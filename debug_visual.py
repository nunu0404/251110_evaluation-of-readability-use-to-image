from __future__ import annotations

import argparse
import torch

from agents_readability import VisualAgent
from vision_encoder import load_vision_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VisualAgent on a single image and show responses.")
    parser.add_argument("image_path", help="Path to rendered code image.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vision_encoder(device=device)
    agent = VisualAgent(model, device=device)
    result = agent.evaluate(args.image_path)
    print("Parsed result:", result)


if __name__ == "__main__":
    main()
