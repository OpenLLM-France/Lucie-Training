# Dataset Card

The Lucie Training Dataset is a curated collection of text data
in English, French, German, Spanish and Italian culled from a variety of sources including: web data, video subtitles, academic papers,
digital books, newspapers, and magazines, some of which were processed by Optical Character Recognition (OCR). It also contains samples of diverse programming languages.

The Lucie Training Dataset was used to pretrain [Lucie-7B](https://huggingface.co/OpenLLM-France/Lucie-7B),
a foundation LLM with strong capabilities in French and English.

## Dataset Description

This dataset was made to provide an extensive and diverse dataset for training Large Language Models (LLMs). Here are some of the principal features of the corpus:
* Data mix:
    * The dataset contains equal amounts of French and English data -- it is in fact one of the biggest collections of French text data that has been preprocessed for LLM training -- with the aim of minimizing anglo-centric cultural biases.
    * German, Spanish and Italian are also represented in small amounts.
    * Code is also included to boost the reasoning capabilities of LLMs.
* Data filtering and deduplication:
    * The dataset has been cleaned in an effort to remove very low-quality data.
    * Duplicate data samples have been removed to some extent, following best practices.
* Ethics:
    * Special care has been taken to respect copyright laws and individual privacy.
      All books, newspapers, monographies, and magazines are in the public domain
  (which depends on the author's date of death and the country of publication).
    * All web data in the dataset came from sites with robots.txt files that do not forbid crawling.

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


### Details on Data Sources

* **Amendements Parlement**
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4) ([nodeputes.fr](http://www.nosdeputes.fr/), [nossenateurs.fr](http://www.nossenateurs.fr/)). [API](https://github.com/regardscitoyens). License: [CC BY-SA](https://www.regardscitoyens.org/#&panel1-2).
  * <u>Description</u>: A collection of proposed amendments by the French parliament: the legal text and description of the requested modification. 
  * <u>Citation</u>: No paper found.

* **American Stories**
  * <u>Source</u>: [dell-research-harvard/AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories). License: [CC BY 4.0](https://huggingface.co/datasets/dell-research-harvard/AmericanStories).
  * <u>Extracted from</u>: [Chronicling America](https://www.loc.gov/collections/chronicling-america/about-this-collection/). License: [Open](https://www.loc.gov/collections/chronicling-america/about-this-collection/rights-and-access/).
  * <u>Description</u>: "The American Stories dataset is a collection of full article texts extracted from historical U.S. newspaper images. It includes nearly 20 million scans from the public domain Chronicling America collection maintained by the Library of Congress. The dataset is designed to address the challenges posed by complex layouts and low OCR quality in existing newspaper datasets" (from the [dataset card](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)).
  * <u>Citation</u>: Melissa Dell, Jacob Carlson, Tom Bryan, Emily Silcock, Abhishek Arora, Zejiang Shen, Luca D'Amico-Wong, Quan Le, Pablo Querubin and Leander Heldring (2023). "American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers," [arxiv:2308.12477](https://arxiv.org/abs/2308.12477v1).


* **Claire (French and English)**
  * <u>Sources</u>:
    * French dataset: [OpenLLM-France/Claire-Dialogue-French-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1).
    * English dataset: [OpenLLM-France/Claire-Dialogue-English-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1).
  * <u>Description</u>: The Claire datasets are composed of transcripts of spoken conversation -- including parliamentary proceedings, interviews, debates, meetings, and free conversations -- as well as some written conversations from theater plays and written chats. The dataset is designed to help downstream performance of models fine-tuned for tasks requiring the comprehension of spontaneous spoken conversation, such as meeting summarization. Each dialogue is split into speech turns, and each speech turn is labeled with the name of the speaker or a unique identifier.
  * <u>Citation</u>: Julie Hunter, Jérôme Louradour, Virgile Rennard, Ismaïl Harrando, Guokan Shang, Jean-Pierre Lorré (2023). The Claire French Dialogue Dataset. [arXiv:2311.16840](https://arxiv.org/abs/2311.16840).

* **Croissant Aligned**
  * <u>Source</u>: [croissantllm/croissant_dataset_no_web_data](https://huggingface.co/datasets/croissantllm/croissant_dataset_no_web_data). License: not specified.
  * <u>Extracted from</u>: [OPUS](https://opus.nlpl.eu/), theses, [song lyrics](https://www.lacoccinelle.net)
  * <u>Description</u>: A collection of English-French translation pairs selected by a custom filtering pipeline. Designed to "improve the multilingual capabilities of the model" ([Arxiv paper](https://arxiv.org/pdf/2402.00786)).
  * <u>Citation</u>: Manuel Faysse, Patrick Fernandes, Nuno M. Guerreiro, António Loison, Duarte M. Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro H. Martins, Antoni Bigata Casademunt, François Yvon, André F.T. Martins, Gautier Viaud, Céline Hudelot, Pierre Colombo (2024). "CroissantLLM: A Truly Bilingual French-English Language Model," [arXiv:2402.00786](https://arxiv.org/abs/2402.00786).

* **Discours Publics** (*)
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>: [Vie Publique](https://www.vie-publique.fr/collection-discours-publics).
  * <u>Description</u>: A collection of public speeches from the principal public actors in France including speeches from the French President starting from 1974 and from the Prime Minister and members of the government starting from 1980.
  * <u>Citation</u>: No paper found.

* **Europarl (monolingual and parallel)**
  * <u>Sources</u>: 
    * `fr-en`, `es-en`, `it-en` parallel data: [Europarl v7](https://www.statmt.org/europarl/v7/). License: [Open](https://www.statmt.org/europarl/).
    * `fr`, `en`, `de`, `es` monolingual data and `de-fr` parallel data: [Europarl v10](https://www.statmt.org/europarl/v10/training-monolingual/). License: [Open](https://www.statmt.org/europarl/).
  * <u>Description</u>: "The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek. The goal of the extraction and processing was to generate sentence aligned text for statistical machine translation systems" ([www.statmt.org](https://www.statmt.org/europarl/)).
  * <u>Citation</u>: Philipp Koehn (2005). "Europarl: A Parallel Corpus for Statistical Machine Translation," MT Summit. 

* **Eurovoc**
  * <u>Source</u>:   [EuropeanParliament/Eurovoc](https://huggingface.co/datasets/EuropeanParliament/Eurovoc). License: [EUPL 1.1](https://joinup.ec.europa.eu/licence/european-union-public-licence-version-11-or-later-eupl).
  * <u>Extracted from</u>: [Cellar](https://op.europa.eu/en/web/cellar). License: [Open](https://op.europa.eu/en/web/cellar).
  * <u>Description</u>: A collection of mutlilingual documents from the data repository of the Publications Office of the European Union annotated with Eurovoc labels. 
  * <u>Citations</u>:
    * Ilias Chalkidis, Emmanouil Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos (2019). "Extreme Multi-Label Legal Text Classification: A Case Study in EU Legislation," Proceedings of the Natural Legal Language Processing Workshop 2019, pages 78–87, Minneapolis, Minnesota. Association for Computational Linguistics.
    * Ilias Chalkidis,  Manos Fergadiotis, Prodromos Malakasiotis and Ion Androutsopoulos (2019). "Large-Scale Multi-Label Text Classification on EU Legislation," Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers).
    * Andrei-Marius Avram, Vasile Pais, and Dan Ioan Tufis (2021). "PyEuroVoc: A Tool for Multilingual Legal Document Classification with EuroVoc Descriptors," Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 92–101, Held Online. INCOMA Ltd.
    * Zein Shaheen, Gerhard Wohlgenannt and Erwin Filtz (2020). "Large scale legal text classification using transformer models," [arXiv:2010.12871](https://arxiv.org/abs/2010.12871v1).

* **FineWebEdu**
  * <u>Source</u>: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
  * <u>Extracted from</u>: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
  * <u>Description</u>: A 1.3 trillion token selection from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), which contains 15 trillion tokens of curated data from 96 Common Crawl dumps. Content in FineWebEdu has been selected by a custom designed classifier for its high-quality, educational content. 
  * <u>Citation</u>: Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," [	arXiv:2406.17557](https://arxiv.org/abs/2406.17557).

* **Gallica Monographies** 
  * <u>Source</u>: Corpus contributed by OpenLLM partners. A version is also published here: [PleIAs/French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books). License: None (public domain).
  * <u>Extracted from</u>: [Gallicagram](https://shiny.ens-paris-saclay.fr/app/gallicagram).
  * <u>Description</u>: A large collection of French monographies in the public domain made available through the French National Library ([Gallica](https://gallica.bnf.fr/accueil/fr/content/accueil-fr?mode=desktop)).
  * <u>Citation</u>: No paper found.

* **Gallica Press**
  * <u>Source</u>: Corpus contributed by OpenLLM partners. A version is also published here: [PleIAs/French-PD-Newspapers](https://huggingface.co/datasets/PleIAs/French-PD-Newspapers). License: None (public domain).
  * <u>Extracted from</u>: [Gallicagram](https://shiny.ens-paris-saclay.fr/app/gallicagram).
  * <u>Description</u>: A large collection of French newspapers and periodicals in the public domain made available through the French National Library ([Gallica](https://gallica.bnf.fr/accueil/fr/content/accueil-fr?mode=desktop)).
  * <u>Citation</u>: No paper found.

* **Gutenberg**
  * <u>Source</u>: Corpus compiled by OpenLLM partners.
  * <u>Extracted from</u>: 
    * [aleph.gutenberg.org](http://aleph.gutenberg.org/) via [Project Gutenberg](https://www.gutenberg.org/). License: [Open](https://www.gutenberg.org/policy/terms_of_use.html).
    * [pgcorpus](https://github.com/pgcorpus/gutenberg). License: [CC BY-4.0](https://zenodo.org/records/2422561).
  * <u>Description</u>: A collection of free eBooks, manually prepared by human annotators. 
  * <u>Citation</u>: No paper found.

* **HAL**
  * <u>Source</u>:
  * <u>Extracted from</u>: [HAL](https://hal.science/).
  * <u>Description</u>: A collection of scientific papers and manuscripts distributed through an open science platform.
  * <u>Citation</u>: 

* **Interventions Parlement**
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4) ([nodeputes.fr](http://www.nosdeputes.fr/), [nossenateurs.fr](http://www.nossenateurs.fr/)). [API](https://github.com/regardscitoyens). License: [CC BY-SA](https://www.regardscitoyens.org/#&panel1-2).
  * <u>Description</u>: Transcripts of speeches made during French parlementary debates.  
  * <u>Citation</u>: No paper found.

* **MathPile**
  * <u>Source</u>: [GAIR/MathPile_Commercial](https://huggingface.co/datasets/GAIR/MathPile_Commercial). License: CC BY-SA 4.0
  * <u>Extracted from</u>: [MathPile](https://huggingface.co/datasets/GAIR/MathPile). License: CC BY-SA-NC 4.0.
  * <u>Description</u>: A preprocessed collection of documents focused on math, including Textbooks, arXiv, Wikipedia, ProofWiki, StackExchange, and web pages from Common Crawl. The content targets a range of levels, from kindergarten through postgraduate level. MathPile_Commercial was obtained by removing documents from MathPile that do not allow commercial use.
  * <u>Citation</u>: Zengzhi Wang, Rui Xia and Pengfei Liu (2023). "Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math," [	arXiv:2312.17120](https://export.arxiv.org/abs/2312.17120).

* **Open Data**
  * <u>Source</u>: [Nicolas-BZRD/DILA_OPENDATA_FR_2023](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main) (balo, dole, inca, kali, legi and sarde subsets). License: ODC-BY.
  * <u>Extracted from</u>: [OpenData](https://echanges.dila.gouv.fr/OPENDATA/) (Data collection date: October, 2023).
  * <u>Description</u>: "The French Government Open Data (DILA) Dataset is a collection of text data extracted from various sources provided by the French government, specifically the Direction de l'information légale et administrative (DILA). This dataset contains a wide range of legal, administrative, and legislative documents. The data has been organized into several categories for easy access and analysis" (from the [dataset card](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main)).
  * <u>Citation</u>: No paper found.

* **Open Edition**
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>: [Open Edition](https://www.openedition.org/).
  * <u>Description</u>: 
  * <u>Citation</u>: No paper found.

* **PeS2o**
  * <u>Source</u>: [allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o). License: 	[ODC BY-v1.0](https://opendatacommons.org/licenses/by/1-0/)
  * <u>Extracted from</u>: [S2ORC](https://github.com/allenai/s2orc) (see [aclanthology](https://aclanthology.org/2020.acl-main.447/)). Knowledge cutoff: 2023-01-03.
  * <u>Description</u>: A preprocessed collection of academic papers designed for pre-training of language models. It includes a subset of full papers and another subset of titles and abstracts.
  * <u>Citation</u>: Luca Soldaini and Kyle Lo (2023). "peS2o (Pretraining Efficiently on S2ORC) Dataset}, Allen Institute for AI. [GitHub](https://github.com/allenai/pes2o).

* **Pile Uncopyrighted**
    * <u>Source</u>: [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted).
    * <u>Extracted from</u>: FreeLaw, StackExchange, USPTO Backgrounds, DM Mathematics, Ubuntu IRC, Phil Papers, NIH ExPorter from [The Pile](https://huggingface.co/datasets/EleutherAI/pile). License: MIT.
    * <u>Description</u> (from the [Datasheet](https://arxiv.org/abs/2201.07311)):
      * FreeLaw: "The Free Law Project is US registered non-profit that provide access to millions of legal opinions and analytical tools for academic studies in the legal realm."
      * StackExchange: "The StackExchange dataset is a dump of anonymized user-contributed content on the Stack Exchange network, a popular collection of websites centered around user-contributed questions and answers."
      * USPTO Backgrounds: "The USPTO Backgrounds dataset is a set of background sections from patents granted by the United States Patent and Trademark Office, derived from its published [bulk archives](https://bulkdata.uspto.gov/)."
      * DM Mathematics: "The DeepMind Mathematics dataset consists of a collection of mathematical problems such as algebra, arithmetic, calculus, number theory, and probability, formatted as natural language prompts [Saxton et al., 2019](https://arxiv.org/abs/1904.01557)."
      * Ubuntu IRC: "The Ubuntu IRC dataset is derived from the publicly available [chatlogs](https://irclogs.ubuntu.com/) of all Ubunturelated channels on the Freenode IRC chat server."
      * PhilPapers: [PhilPapers](https://philpapers.org/) is a dataset of open access philosophy publications from an international database maintained by the Center for Digital Philosophy at the University of Western Ontario.
      * NIH ExPORTER: "The NIH Grant abstracts provides a bulk-data repository for awarded applications through the ExPORTER4 service covering the fiscal years 1985-present."
    * <u>Citation</u>:
      * Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy (2020). "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," [	arXiv:2101.00027](https://arxiv.org/abs/2101.00027).
      * Stella Biderman, Kieran Bicheno, Leo Gao (2022). "Datasheet for the Pile," [	arXiv:2201.07311](https://arxiv.org/abs/2201.07311).

* **Questions Ecrites Parlement**
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4) ([text](https://data.regardscitoyens.org/nosdeputes.fr/)). License: [CC BY-NC-SA](https://data.regardscitoyens.org/nosdeputes.fr/).
  * <u>Description</u>: Collection of long written questions, read during a session at the french national assembly: from a member of french parliament to a minister  (Minister got 2 month to respond). ([text](https://data.regardscitoyens.org/nosdeputes.fr/)).
  * <u>Citation</u>: No paper found.

* **RedPajama (v2)**
  * <u>Source</u>: [togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2). License: Apache 2.0 (data preparation code), Not specified (data) but see [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).
  * <u>Description</u>: "RedPajama-V2 is an open dataset for training large language models. The dataset includes over 100B text documents coming from 84 CommonCrawl snapshots and processed using the [CCNet](https://github.com/facebookresearch/cc_net) pipeline. Out of these, there are 30B documents in the corpus that additionally come with quality signals, and 20B documents that are deduplicated" (from [GitHub](https://github.com/togethercomputer/RedPajama-Data)).
  * <u>Citation</u>: Together Computer (2023). "RedPajama-Data-v2: an Open Dataset with 30 Trillion Tokens for Training Large Language Models," [GitHub](https://github.com/togethercomputer/RedPajama-Data).

* **STAC**
  * <u>Source</u>: [STAC](https://www.irit.fr/STAC/corpus.html). License: CC BY-SA-NC 4.0.
  * <u>Description</u>: A collection of chats from an online version of the game Settlers of Catan.
  * <u>Citation</u>: Nicholas Asher, Julie Hunter, Mathieu Morey, Farah Benamara and Stergos Afantenos (2016). "Discourse structure and dialogue acts in multiparty dialogue: the STAC corpus," The Tenth International Conference on Language Resources and Evaluation (LREC 2016). European Language Resources Association, pp. 2721-2727.

* **The Stack**
  * <u>Source</u>: [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup). License: Other (mixture of copyleft licenses).
  * <u>Description</u>: "The Stack contains over 6TB of permissively-licensed source code files covering 358 programming languages. The dataset was created as part of the BigCode Project, an open scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs). The Stack serves as a pre-training dataset for Code LLMs, i.e., code-generating AI systems which enable the synthesis of programs from natural language descriptions as well as other from code snippets. This is the near-deduplicated version with 3TB data" (from the [dataset card](https://huggingface.co/datasets/bigcode/the-stack-dedup)).
  * <u>Citation</u>: Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra and Harm de Vries (2022). "The Stack: 3 TB of permissively licensed source code," [arxiv:2211.15533](https://arxiv.org/abs/2211.15533).

* **Thèses**
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>: [theses.fr](https://theses.fr/?domaine=theses) and  [HAL???]().
  * <u>Description</u>: 
  * <u>Citation</u>: No paper found.

* **Wikipedia, Wikisource, Wiktionary**
  * <u>Source</u>: Corpus contributed by OpenLLM partners. Also published here: [OpenLLM-France/wikipedia](https://huggingface.co/datasets/OpenLLM-France/wikipedia).
  * <u>Extracted from</u>: [Wikimedia dumps](https://dumps.wikimedia.org/other/enterprise_html/runs/). License: [GFDL/CC BY-SA](https://dumps.wikimedia.org/legal.html)
  * <u>Description</u>:
  * <u>Citation</u>: No paper found.

* **YouTube**
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>:
  * <u>Description</u>: 
  * <u>Citation</u>: No paper found.

