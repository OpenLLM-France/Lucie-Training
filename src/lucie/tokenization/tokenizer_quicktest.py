import tiktoken
import transformers
from tokenizer_train import test_tokenizer


def norm_for_display(s):
    return s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def norm_spaces(s):
    return s.replace("\u00A0", " ").replace("\r", "")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a tokenizer.")
    parser.add_argument(
        "tokenizer",
        help="Tokenizer to evaluate",
    )
    args = parser.parse_args()

    example_sentences = [
        # "Mais? Mais?\nMais?\tMais?   Mais?\n|Mais? (Mais?)",
        # "en ? en ?\r\nen ?\ten ?   en ? (en ?)\n(en ?)",
        # (
        #     "   [INST] Coucou [/INST] Hello. Hello . [INST]      \n"
        #     "Mais en Français, comment est-ce que ça se passera ?"
        #     " \t\t\n\n1999, 2000, 2002.2, 1 000 000, 1\u00A0000\u00A0000"
        #     "[/INST] Eh bien ␣ alors あ゙"
        # ),
        # Unknown characters
        "␣ あ゙",
        # Digits
        "$1 234\u00A0567,890€ 1.2 3/4",
        # Spaces
        "Mot Mot\nMot\n Mot\n   Mot",
        # Spaces & Brackets
        "(Mot) (Mot)\n(Mot)\n (Mot)\n" + " " * 13 + "(Mot)",
        "[INST]",
        # Punctuations
        "website.fr là. (etc...) .- -.",
    ]

    if args.tokenizer.lower() in ["gpt-4"]:
        tokenizer = tiktoken.encoding_for_model(args.tokenizer.lower())
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    for example_sentence in example_sentences:
        tokens, decoded = test_tokenizer(tokenizer, example_sentence)

        print("-" * 50)
        print("* Reference.........:", norm_for_display(example_sentence))
        # print("* Decoded:", norm_for_display(decoded))
        has_bos = decoded.startswith("<")
        has_eos = decoded.endswith(">")
        if has_bos:
            decoded = decoded[decoded.index(">") + 1 :]
        if has_eos:
            while decoded[-1] != "<":
                decoded = decoded[:-1]
            decoded = decoded[:-1]
        bos_eos_details = f"{' no 'if has_bos or has_eos else ''}{'BOS' if has_bos else ''}{'/EOS' if has_eos else ''}"
        print(
            f"* Decoded{bos_eos_details}:",
            norm_for_display(decoded),
        )
        print("* Tokens: ", tokens)

        decoded = decoded.lstrip()
        example_sentence = example_sentence.lstrip()
        # print("* OK 100%.....................:", example_sentence == decoded)
        example_sentence = norm_spaces(example_sentence)
        decoded = norm_spaces(decoded)
        # print("* OK up to space normalization:", example_sentence == decoded)
        if example_sentence != decoded:
            print("KO!!!")
            for i, (a, b) in enumerate(zip(example_sentence, decoded)):
                if a != b:
                    print(f"  {i}: '{a}' ({ord(a)}) != '{b}' ({ord(b)})")
                    break
