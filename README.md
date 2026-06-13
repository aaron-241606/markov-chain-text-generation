# Task-03: Text Generation with Markov Chains

> Implement a simple text generation algorithm using Markov chains — a statistical model that predicts the probability of a character or word based on the previous one(s).

---

## 📌 Overview

This project builds a Markov Chain text generator from scratch using pure Python. Given any training corpus, the model learns transition probabilities between words/characters and generates new text that mimics the style of the input.

---

## 🗂️ Project Structure

```
markov-chain-text-generation/
├── markov.py             # Core Markov Chain model (word & character level)
├── generate.py           # CLI text generation script
├── train_and_generate.py # Full pipeline: train on corpus + generate
├── data/
│   └── sample_corpus.txt # Sample training text
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/aaron-241606/markov-chain-text-generation.git
cd markov-chain-text-generation
pip install -r requirements.txt
```

No ML libraries needed — runs on pure Python!

---

## 🚀 Usage

### Generate text from sample corpus
```bash
python generate.py --corpus data/sample_corpus.txt --order 2 --words 100
```

### Character-level generation
```bash
python generate.py --corpus data/sample_corpus.txt --mode char --order 3 --words 200
```

### Custom seed word
```bash
python generate.py --corpus data/sample_corpus.txt --seed "the future" --words 150
```

---

## 🧠 How Markov Chains Work

```
Training:  "the cat sat" → P("sat" | "the cat") = 1.0
           "the cat ran" → P("ran" | "the cat") = 0.5, P("sat" | "the cat") = 0.5

Generation: Start with seed → sample next word by probability → repeat
```

| Parameter | Description                              |
|-----------|------------------------------------------|
| Order     | Number of previous words/chars to consider |
| Mode      | `word` (default) or `char` level         |
| Seed      | Starting text for generation             |

---

## 📚 References

- [#1 Markov Chains Explained](https://en.wikipedia.org/wiki/Markov_chain)
- [#2 Text Generation with Markov Chains](https://towardsdatascience.com/text-generation-with-markov-chains)

---

## 🏢 Credits

**Prodigy Infotech** – Task-03 Internship Project
