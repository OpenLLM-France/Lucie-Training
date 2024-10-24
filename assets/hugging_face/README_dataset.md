# Dataset Card

The Lucie Training Dataset is a curated collection of text data
in English, French, German, Spanish and Italian,
from the web,
video subtitles,
collections of books, newspapers, monographies, and magazines processed by Optical Character Recognition (OCR),
as well as collections of files in diverse programming languages.

It was used to pretrain [Lucie-7B](https://huggingface.co/OpenLLM-France/Lucie-7B),
a foundation LLM with strong capabilities in French and English.

## Dataset Description

This dataset was made to provide an extensive and diverse dataset for training Large Language Models (LLM),
with the following motivations in mind:
* Data mix:
    * French is as well represented as English
    (Lucie Training Dataset is one of the biggest of collection of French text data with a minimum of quality),
    to avoid that the LLM is culturally biased towards English.
    * German, Spanish and Italian are also represented to some extend,
    * Code is also included to boost the reasoning capabilities of LLM.
* Data filtering and deduplication:
    * The dataset is cleaned low-quality data
    * The dataset is cleaned from duplicates to some extend, following best practices.
* Ethics:
    * A special care was taken to respect copyright laws and the privacy of individuals.
      All books, newspapers, monographies, and magazines are in the public domain
  (which depends on the author's death date, and the country of publication).
    * There is no data from the web for which robots.txt files forbid crawling.

### Dataset Structure

The corpus contains the following information for each text sample:
* `text`: the text sample itself.
* `source`: an identifier for the source(s) of the text sample (`Wikipedia`, `RedPajama`, `Gutenberg`, …).
  The list of all sources is described in this document.
* `id`: an identifier that is unique among the source.
* `language`: the language of the text sample, which can be:
    * the ISO 639-1 code of a natural language: `en`, `fr`, `de`, `es`, or `it`;
    * the common name prefixed by "`code:`" of a programming language:  `code:python`, `code:c++`, …; or
    * a list of ISO 639-1 codes separated by commas, if the text sample is multilingual: `fr,en`, `de,fr`, `es,en`, `it,en`
     (or in the opposite order if the languages appear in the opposite order in the text).
* `url` (optional): the URL of the original text sample on the web, if available.
* `title` (optional): the title of the original text sample, if available.
* `author` (optional): the author of the original text sample, if available.
   Usually the author name in plain text, except for `Gutenberg` where it is the JSON serialized object of the author metadata.
* `date` (optional): the publication date of the original text sample, if available. The text format of the source depends on the source.
* `quality_signals` (optional): a list of quality signals about the text sample (that could be used for further filtering or sample weighting).
  It can include indicators computed by `fasttext` and `CCNet`, statistics about occurrences of characters, words, special characters, etc.
  This field is always a JSON serialized object.
* `extra` (optional): JSON serialized extra information about the text sample.
  This can include metadata about the source subset, the rights, etc.

Examples of metadata (except from `text`) are shown for each source in [metadata_examples.json](metadata_examples.json).

### Example use in python

Load the dataset using the `datasets` library:
```python
from datasets import load_dataset

kwargs = {"split": "train", "streaming": True}

dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", **kwargs)
```

Several configurations are available to select a language, a source, or both, illustrated in the following examples.

Load data in French:
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "fr", **kwargs)
```
Load data where French and English are aligned:
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "fr,en", **kwargs)
```
Load data corresponding to files with programming languages:
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "code", **kwargs)
```
Load data in Python:
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "code:python", **kwargs)
```
Load data from Wikipedia (in available languages):
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Wikipedia", **kwargs)
```
Load data from French pages of Wikipedia ([wikipedia.fr](https://www.wikipedia.fr/)):
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Wikipedia-fr", **kwargs)
```
