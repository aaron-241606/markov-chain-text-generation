"""
Task-03: Markov Chain Text Generation Model
Prodigy Infotech Internship
"""

import random
import re
from collections import defaultdict


class MarkovChain:
    """
    A Markov Chain text generator supporting both word-level and
    character-level generation with configurable order (n-gram size).
    """

    def __init__(self, order=2, mode="word"):
        """
        Args:
            order: Number of previous tokens to condition on (n-gram size).
            mode:  'word' for word-level, 'char' for character-level.
        """
        self.order = order
        self.mode  = mode
        self.chain = defaultdict(list)
        self.start_tokens = []

    def _tokenize(self, text):
        """Split text into tokens based on mode."""
        if self.mode == "word":
            tokens = re.findall(r"\b\w[\w']*\b|[.,!?;]", text.lower())
        else:
            tokens = list(text)
        return tokens

    def train(self, text):
        """Build the Markov chain from a training corpus."""
        tokens = self._tokenize(text)
        if len(tokens) < self.order + 1:
            raise ValueError(f"Corpus too short. Need at least {self.order + 1} tokens.")

        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i: i + self.order])
            next_token = tokens[i + self.order]
            self.chain[state].append(next_token)

        # Collect valid starting states (beginning of sentences)
        self.start_tokens = [
            tuple(tokens[i: i + self.order])
            for i in range(len(tokens) - self.order)
            if i == 0 or tokens[i - 1] in ".!?"
        ]

        print(f"✅ Trained on {len(tokens)} tokens | {len(self.chain)} unique states | order={self.order}")

    def generate(self, num_tokens=100, seed=None):
        """
        Generate text using the trained Markov chain.

        Args:
            num_tokens: Number of tokens to generate.
            seed:       Optional starting state (tuple of `order` tokens).
        Returns:
            Generated text as a string.
        """
        if not self.chain:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        if seed:
            seed_tokens = self._tokenize(seed)
            state = tuple(seed_tokens[-self.order:])
            if state not in self.chain:
                print(f"⚠️  Seed state not found in chain, using random start.")
                state = random.choice(list(self.chain.keys()))
        else:
            state = random.choice(self.start_tokens or list(self.chain.keys()))

        output = list(state)

        for _ in range(num_tokens - self.order):
            if state not in self.chain:
                state = random.choice(list(self.chain.keys()))
                output.extend(state)
            next_token = random.choice(self.chain[state])
            output.append(next_token)
            state = tuple(output[-self.order:])

        if self.mode == "word":
            return self._detokenize_words(output)
        else:
            return "".join(output)

    def _detokenize_words(self, tokens):
        """Join word tokens back into readable text."""
        result = []
        for i, token in enumerate(tokens):
            if token in ".,!?;:" or i == 0:
                result.append(token)
            else:
                result.append(" " + token)
        text = "".join(result)
        # Capitalize first letter and after sentence-ending punctuation
        text = re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), text)
        return text.strip()

    def get_stats(self):
        """Print statistics about the trained model."""
        if not self.chain:
            print("Model not trained yet.")
            return
        avg_transitions = sum(len(v) for v in self.chain.values()) / len(self.chain)
        print(f"\n📊 Model Statistics:")
        print(f"   Order           : {self.order}")
        print(f"   Mode            : {self.mode}")
        print(f"   Unique states   : {len(self.chain)}")
        print(f"   Avg transitions : {avg_transitions:.2f} per state")
        print(f"   Start states    : {len(self.start_tokens)}")
