import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sampler import sample
from src.extractor import extract

if __name__ == "__main__":
    input_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "sampled")

    # Step 1: Sample the graph
    sample(
        input_dir=input_dir,
        output_dir=output_dir,
        target_users=20000,
        seed_ratio=0.1,
        hub_ratio=0.7,
        bot_ratio=0.14,
    )

    # Step 2: Extract the sampled data
    extract(input_dir=input_dir, output_dir=output_dir)
