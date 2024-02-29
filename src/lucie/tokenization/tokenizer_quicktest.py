import tiktoken
import transformers
from tokenizer_train import test_tokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a tokenizer.")
    parser.add_argument(
        "tokenizer",
        help="Tokenizer to evaluate",
    )
    args = parser.parse_args()

    example_sentences = [
        "Mais? Mais?",
        "en? en?",
        (
            "   [INST] Coucou [/INST] Hello. Hello . [INST]      \n"
            "Mais en Français, comment est-ce que ça se passera ?"
            " \t\t\n\n1999, 2000, 2002.2, 1 000 000, 1\u00A0000\u00A0000"
            "[/INST] Eh bien ␣ alors あ゙"
        ),
    ]

    if args.tokenizer in ["gpt-4"]:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    for example_sentence in example_sentences:
        tokens, decoded = test_tokenizer(tokenizer, example_sentence)

        print("* Tokens:", tokens)
        print("* Decoded:", decoded)

        if decoded.startswith("<"):
            decoded = decoded[decoded.index(">") + 1 :]
        if decoded.endswith(">"):
            while decoded[-1] != "<":
                decoded = decoded[:-1]
            decoded = decoded[:-1]
        print("* Decoded no BOS/EOS:", decoded.replace("\n", "\\n"))
        print("* Reference.........:", example_sentence.replace("\n", "\\n"))

        decoded = decoded.lstrip()
        example_sentence = example_sentence.lstrip()
        print("* OK 100%:", example_sentence == decoded)
        example_sentence = example_sentence.replace("\u00A0", " ")
        print("* OK up to space:", example_sentence == decoded)
        if example_sentence != decoded:
            for i, (a, b) in enumerate(zip(example_sentence, decoded)):
                if a != b:
                    print(f"  {i}: '{a}' ({ord(a)}) != '{b}' ({ord(b)})")
                    break
