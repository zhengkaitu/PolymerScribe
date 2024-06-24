import pandas as pd
import re
from polymerscribe.tokenizer import Tokenizer


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
    df = pd.read_csv("data/polyBERT_len85_0.csv")
    df.columns = ["index", "bigsmiles"]

    df["tokenized_bigsmiles"] = df["bigsmiles"].apply(tokenize_by_regex)
    df["file_path"] = df["index"].apply(
        lambda i: f"polyBERT_len85_0/polyBERT_len85_0_{i}.svg"
    )
    df.to_csv("data/polyBERT_len85_0_tokenized.csv", index=False)


def make_vocab_file() -> None:
    df = pd.read_csv("data/polyBERT_len85_0_tokenized.csv")
    texts = df["tokenized_bigsmiles"].tolist()

    tokenizer = Tokenizer(path=None)
    tokenizer.fit_on_texts(texts)
    tokenizer.save("polymerscribe/vocab/vocab_polybert.json")


def sample_and_split() -> None:
    df = pd.read_csv("data/polyBERT_len85_0_tokenized.csv")
    train_df = df.sample(frac=0.8, random_state=200)
    remaining_df = df.drop(train_df.index)
    valid_df = remaining_df.sample(frac=0.5, random_state=200)
    test_df = remaining_df.drop(valid_df.index)

    train_df.to_csv("data/polyBERT_len85_0_tokenized_train.csv", index=False)
    valid_df.to_csv("data/polyBERT_len85_0_tokenized_valid.csv", index=False)
    test_df.to_csv("data/polyBERT_len85_0_tokenized_test.csv", index=False)


def main() -> None:
    tokenize_csv_by_regex()
    make_vocab_file()
    sample_and_split()


if __name__ == "__main__":
    main()
