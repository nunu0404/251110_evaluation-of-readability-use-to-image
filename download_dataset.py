from datasets import load_dataset
ds = load_dataset("se2p/code-readability-merged")
ds.save_to_disk("code_readability_merged_dataset")