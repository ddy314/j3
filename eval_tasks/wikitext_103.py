"""WikiText-103 preprocessing for the GIBC evaluation task."""

from __future__ import annotations

import re


def wikitext_detokenizer(doc: dict[str, str]) -> str:
    """Match the public WikiText detokenization used by lm-evaluation-harness."""

    string = doc["text"]
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    return string.replace(" 's", "'s")


def process_results(doc: dict[str, str], results: tuple[float]) -> dict[str, tuple[float, int]]:
    """Report the same word/byte metrics as the standard harness task."""

    (loglikelihood,) = results
    words = len(re.split(r"\s+", doc["text"]))
    byte_count = len(doc["text"].encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, words),
        "byte_perplexity": (loglikelihood, byte_count),
        "bits_per_byte": (loglikelihood, byte_count),
    }
