"""
Task-03: Generate text using a trained Markov Chain
Prodigy Infotech Internship
"""

import argparse
from markov import MarkovChain


def main(corpus_path, order, mode, num_words, seed):
    print(f"📂 Loading corpus: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"   Characters : {len(text):,}")
    print(f"   Mode       : {mode}-level | Order: {order}\n")

    model = MarkovChain(order=order, mode=mode)
    model.train(text)
    model.get_stats()

    print(f"\n{'='*60}")
    print("📝 Generated Text:")
    print('='*60 + "\n")

    generated = model.generate(num_tokens=num_words, seed=seed)
    print(generated)
    print(f"\n{'='*60}")
    print(f"   Tokens generated: {num_words}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a Markov Chain")
    parser.add_argument("--corpus", type=str,  default="data/sample_corpus.txt", help="Path to training corpus")
    parser.add_argument("--order",  type=int,  default=2,    help="Markov chain order (n-gram size)")
    parser.add_argument("--mode",   type=str,  default="word", choices=["word", "char"], help="Tokenization mode")
    parser.add_argument("--words",  type=int,  default=100,  help="Number of tokens to generate")
    parser.add_argument("--seed",   type=str,  default=None, help="Seed phrase to start generation")
    args = parser.parse_args()

    main(args.corpus, args.order, args.mode, args.words, args.seed)
