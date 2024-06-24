import cairosvg
import glob
import multiprocessing
import os
import pandas as pd
import re
import time
import traceback as tb
from polymerscribe.tokenizer import Tokenizer
from tqdm import tqdm
from typing import Tuple


def tokenize_by_regex(bigsmiles: str) -> str:
    """
        Tokenize a BigSMILES molecule or reaction,
        adapted from the SMILES tokenizer in https://github.com/pschwllr/MolecularTransformer
    """
    # pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|<|>|{|}|,|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(bigsmiles)]

    # ensuring no token was dropped
    assert bigsmiles == "".join(tokens), f"bigsmiles: {bigsmiles}, tokens: {tokens}"

    tokenized_bigsmiles = " ".join(tokens)

    return tokenized_bigsmiles


def tokenize_csv_by_regex() -> None:
    print("tokenize_csv_by_regex")

    df = pd.read_csv("data/polyBERT_len85_0.csv")
    df.columns = ["index", "bigsmiles"]

    df["tokenized_bigsmiles"] = df["bigsmiles"].apply(tokenize_by_regex)
    df["file_path"] = df["index"].apply(
        lambda i: f"polyBERT_len85_0/png/polyBERT_len85_0_{i}.png"
    )
    df.to_csv("data/polyBERT_len85_0_tokenized.csv", index=False)


def make_vocab_file() -> None:
    print("make_vocab_file")

    df = pd.read_csv("data/polyBERT_len85_0_tokenized.csv")
    texts = df["tokenized_bigsmiles"].tolist()

    tokenizer = Tokenizer(path=None)
    tokenizer.fit_on_texts(texts)
    tokenizer.save("polymerscribe/vocab/vocab_polybert.json")


def sample_and_split() -> None:
    print("sample_and_split")

    df = pd.read_csv("data/polyBERT_len85_0_tokenized.csv")
    df = df[:20000]

    train_df = df.sample(frac=0.8, random_state=200)
    remaining_df = df.drop(train_df.index)
    valid_df = remaining_df.sample(frac=0.5, random_state=200)
    test_df = remaining_df.drop(valid_df.index)

    train_df.to_csv("data/polyBERT_len85_0_tokenized_train.csv", index=False)
    valid_df.to_csv("data/polyBERT_len85_0_tokenized_valid.csv", index=False)
    test_df.to_csv("data/polyBERT_len85_0_tokenized_test.csv", index=False)


def _rasterize_svg_helper(_args: Tuple[int, str]) -> None:
    i, image_file = _args
    if i > 0 and i % 1000 == 0:
        print(f"Rasterizing {i}th image")

    output_file = os.path.basename(image_file)
    output_file = f"{output_file[:-4]}.png"
    output_file = os.path.join("data/polyBERT_len85_0/png", output_file)

    try:
        cairosvg.svg2png(url=image_file, write_to=output_file)
    except Exception as e:
        # tb.print_exc()
        print(f"Cannot rasterize image file {image_file}!")


def rasterize_svg():
    print("rasterize_svg")

    os.makedirs("data/polyBERT_len85_0/png", exist_ok=True)
    max_count = 100000

    image_files = glob.glob("data/polyBERT_len85_0/svg/*.svg")
    image_files = image_files[:max_count]

    p = multiprocessing.Pool()
    p.imap_unordered(_rasterize_svg_helper, enumerate(image_files))

    p.close()
    p.join()


def main() -> None:
    tokenize_csv_by_regex()
    make_vocab_file()
    sample_and_split()
    rasterize_svg()


if __name__ == "__main__":
    main()
