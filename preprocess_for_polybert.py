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
    df["tokenized_bigsmiles"] = df["0"].apply(tokenize_by_regex)
    df.to_csv("data/polyBERT_len85_0_tokenized.csv", index=False)


def make_vocab_file() -> None:
    df = pd.read_csv("data/polyBERT_len85_0_tokenized.csv")
    texts = df["tokenized_bigsmiles"].tolist()

    tokenizer = Tokenizer(path=None)
    tokenizer.fit_on_texts(texts)
    tokenizer.save("polymerscribe/vocab/vocab_polybert.json")


def main() -> None:
    tokenize_csv_by_regex()
    make_vocab_file()


if __name__ == "__main__":
    main()
