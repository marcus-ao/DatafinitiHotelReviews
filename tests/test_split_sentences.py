import importlib.util
import unittest
from pathlib import Path


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


split_mod = load_module("scripts/03_split_sentences.py", "split_mod")


class Sent:
    def __init__(self, text: str):
        self.text = text


class DummyDoc:
    def __init__(self, sents):
        self.sents = [Sent(text) for text in sents]


class DummyNLP:
    def __call__(self, _text: str):
        return DummyDoc(["Great location.", "Very", "close to downtown."])


class SplitSentencesTestCase(unittest.TestCase):
    def test_split_review_merges_short_fragment_before_filtering(self):
        sentences = split_mod.split_review(
            "dummy",
            min_len=10,
            merge_len=15,
            nlp=DummyNLP(),
        )
        self.assertEqual(sentences, ["Great location. Very", "close to downtown."])


if __name__ == "__main__":
    unittest.main()
