import argparse
import os
import shutil

import torch
from transformers import AutoModel


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge and save a PEFT model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to the input model directory'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path where the merged model will be saved'
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    if not args.input_path.endswith("/"):
        args.input_path += "/"  # Trailing slash forces transformers to use only files in the directory
    
    # Load and merge the model
    print(f"Loading model from {args.input_path}...")
    model = AutoModel.from_pretrained(
        args.input_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    print("Merging model layers...")
    model.language_model = model.language_model.merge_and_unload()
    
    # Create output directory and save model
    print(f"Saving merged model to {args.output_path}...")
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)
    model.save_pretrained(args.output_path)
    
    print("Model successfully merged and saved!")

if __name__ == "__main__":
    main()
