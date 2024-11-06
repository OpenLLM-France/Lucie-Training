# Dataset Card

The Lucie Training Dataset is a curated collection of text data
in English, French, German, Spanish and Italian culled from a variety of sources including: web data, video subtitles, academic papers,
digital books, newspapers, and magazines, some of which were processed by Optical Character Recognition (OCR). It also contains samples of diverse programming languages.

The Lucie Training Dataset was used to pretrain [Lucie-7B](https://huggingface.co/OpenLLM-France/Lucie-7B),
a foundation LLM with strong capabilities in French and English.

Table of Contents:
* [Dataset Description](#dataset-description)
  * [Dataset Structure](#dataset-structure)
  * [Dataset Composition](#dataset-composition)
    * [Web](#web)
    * [Newspaper](#newspaper)
    * [Technical](#technical)
    * [Book](#book)
    * [Parallel Corpora](#parallel-corpora)
    * [Legislative Texts](#legislative-texts)
    * [Wiki](#wiki)
    * [Math](#math)
    * [Forum](#forum)
    * [Dialogue](#dialogue)
    * [Legislative Transcripts](#legislative-transcripts)
    * [Programming](#programming)
  * [Details on Data Sources](#details-on-data-sources)
    <!-- * [RedPajama (v2)](#redpajama-v2) -->
* [Example use in python](#example-use-in-python)
* [License](#license)
* [Citation](#citation)
* [Contact](#contact)


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


### Dataset Composition

<table>
<thead>
<tr>
  <th><a href="#subset"><strong>subset</strong></a></th>
  <th><strong>language</strong></th>
  <th><strong>M docs</strong></th>
  <th><strong>B words</strong></th>
  <th><strong>B tokens</strong></th>
  <th><strong>B chars</strong></th>
  <th></th>
</tr>
</thead>
<tbody>
<tr>
  <td colspan="7"><h4 id=web>Web</h4></td></tr>
<tr>
  <td><a href="#redpajama-v2"><strong>RedPajama</strong></a></td>
  <td><strong>fr</strong></td>
  <td>640.77</td>
  <td>477.758</td>
  <td>741.023</td>
  <td>2974.596</td>
  <td><strong>2014</strong> (1.32 B words), <strong>2015</strong> (0.776 B words), <strong>2016</strong> (2.033 B words), <strong>2017</strong> (55.665 B words), <strong>2018</strong> (81.345 B words), <strong>2020</strong> (75.141 B words), <strong>2021</strong> (82.439 B words), <strong>2022</strong> (64.866 B words), <strong>2023</strong> (27.239 B words)</td>
</tr>
<tr>
  <td><a href="#redpajama-v2"><strong>RedPajama</strong></a></td>
  <td><strong>de</strong></td>
  <td>162.779</td>
  <td>103.078</td>
  <td>201.371</td>
  <td>747.631</td>
  <td><strong>2021</strong> (17.591 B words), <strong>2023</strong> (24.704 B words)</td>
</tr>
<tr>
  <td><a href="#redpajama-v2"><strong>RedPajama</strong></a></td>
  <td><strong>es</strong></td>
  <td>169.447</td>
  <td>121.751</td>
  <td>197.125</td>
  <td>746.984</td>
  <td><strong>2021</strong> (20.821 B words), <strong>2023</strong> (28.868 B words)</td>
</tr>
<tr>
  <td><a href="#redpajama-v2"><strong>RedPajama</strong></a></td>
  <td><strong>it</strong></td>
  <td>97.324</td>
  <td>60.194</td>
  <td>108.416</td>
  <td>393.012</td>
  <td><strong>2021</strong> (10.266 B words), <strong>2023</strong> (14.403 B words)</td>
</tr>
<tr>
  <td><a href="#finewebedu"><strong>FineWebEdu</strong></a></td>
  <td><strong>en</strong></td>
  <td>421.209</td>
  <td>327.453</td>
  <td>467.837</td>
  <td>2018.215</td>
  <td><strong>2019</strong> (65.275 B words), <strong>2020</strong> (59.076 B words), <strong>2022</strong> (58.788 B words), <strong>2023</strong> (62.085 B words), <strong>2024</strong> (9.197 B words)</td>
</tr>
<tr>
  <td colspan="7"><h4 id=newspaper>Newspaper</h4></td></tr>
<tr>
  <td><a href="#gallicapress"><strong>GallicaPress</strong></a></td>
  <td><strong>fr</strong></td>
  <td>3.205</td>
  <td>67.496</td>
  <td>121.606</td>
  <td>408.882</td>
  <td></td>
</tr>
<tr>
  <td><a href="#americanstories"><strong>AmericanStories</strong></a></td>
  <td><strong>en</strong></td>
  <td>59.42</td>
  <td>8.902</td>
  <td>14.313</td>
  <td>50.844</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=technical>Technical</h4></td></tr>
<tr>
  <td><a href="#pes2o"><strong>PeS2o</strong></a></td>
  <td><strong>en</strong></td>
  <td>38.972</td>
  <td>42.296</td>
  <td>65.365</td>
  <td>268.963</td>
  <td></td>
</tr>
<tr>
  <td><a href="#hal"><strong>HAL</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.349</td>
  <td>9.356</td>
  <td>16.224</td>
  <td>58.308</td>
  <td></td>
</tr>
<tr>
  <td><a href="#theses"><strong>Theses</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.102</td>
  <td>7.547</td>
  <td>14.06</td>
  <td>47.758</td>
  <td></td>
</tr>
<tr>
  <td><a href="#persee"><strong>Persee</strong></a></td>
  <td><strong>fr</strong></td>
  <td>1.094</td>
  <td>3.25</td>
  <td>5.754</td>
  <td>20.314</td>
  <td></td>
</tr>
<tr>
  <td><a href="#pile-uncopyrighted"><strong>Pile (USPTO_Backgrounds)</strong></a></td>
  <td><strong>en</strong></td>
  <td>5.139</td>
  <td>3.492</td>
  <td>5.105</td>
  <td>22.309</td>
  <td></td>
</tr>
<tr>
  <td><a href="#openedition"><strong>OpenEdition</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.939</td>
  <td>2.225</td>
  <td>3.604</td>
  <td>14.459</td>
  <td></td>
</tr>
<tr>
  <td><a href="#pile-uncopyrighted"><strong>Pile (PhilPapers)</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.031</td>
  <td>0.363</td>
  <td>0.618</td>
  <td>2.304</td>
  <td></td>
</tr>
<tr>
  <td><a href="#pile-uncopyrighted"><strong>Pile (NIH_ExPorter)</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.914</td>
  <td>0.288</td>
  <td>0.431</td>
  <td>1.979</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=book>Book</h4></td></tr>
<tr>
  <td><a href="#gallicamonographies"><strong>GallicaMonographies</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.278</td>
  <td>15.106</td>
  <td>25.169</td>
  <td>90.456</td>
  <td></td>
</tr>
<tr>
  <td><a href="#gutenberg"><strong>Gutenberg</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.056</td>
  <td>3.544</td>
  <td>5.516</td>
  <td>20.579</td>
  <td></td>
</tr>
<tr>
  <td><a href="#gutenberg"><strong>Gutenberg</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.003</td>
  <td>0.227</td>
  <td>0.383</td>
  <td>1.392</td>
  <td></td>
</tr>
<tr>
  <td><a href="#gutenberg"><strong>Gutenberg</strong></a></td>
  <td><strong>de</strong></td>
  <td>0.002</td>
  <td>0.099</td>
  <td>0.193</td>
  <td>0.654</td>
  <td></td>
</tr>
<tr>
  <td><a href="#gutenberg"><strong>Gutenberg</strong></a></td>
  <td><strong>it</strong></td>
  <td>0.001</td>
  <td>0.066</td>
  <td>0.129</td>
  <td>0.414</td>
  <td></td>
</tr>
<tr>
  <td><a href="#gutenberg"><strong>Gutenberg</strong></a></td>
  <td><strong>es</strong></td>
  <td>0.001</td>
  <td>0.051</td>
  <td>0.092</td>
  <td>0.303</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=parallel-corpora>Parallel Corpora</h4></td></tr>
<tr>
  <td><a href="#croissantaligned"><strong>CroissantAligned</strong></a></td>
  <td><strong>fr-en</strong></td>
  <td>408.029</td>
  <td>16.911</td>
  <td>25.351</td>
  <td>107.003</td>
  <td></td>
</tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>EuroparlAligned</strong></a></td>
  <td><strong>it-en</strong></td>
  <td>1.901</td>
  <td>0.1</td>
  <td>0.151</td>
  <td>0.638</td>
  <td></td>
</tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>EuroparlAligned</strong></a></td>
  <td><strong>fr-en</strong></td>
  <td>2.003</td>
  <td>0.105</td>
  <td>0.143</td>
  <td>0.655</td>
  <td></td>
</tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>EuroparlAligned</strong></a></td>
  <td><strong>es-en</strong></td>
  <td>1.961</td>
  <td>0.103</td>
  <td>0.143</td>
  <td>0.631</td>
  <td></td>
</tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>EuroparlAligned</strong></a></td>
  <td><strong>de-fr</strong></td>
  <td>1.792</td>
  <td>0.091</td>
  <td>0.141</td>
  <td>0.621</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=legislative-texts>Legislative Texts</h4></td></tr>
<tr>
  <td><a href="#pile-uncopyrighted"><strong>Pile (FreeLaw)</strong></a></td>
  <td><strong>en</strong></td>
  <td>3.415</td>
  <td>8.204</td>
  <td>14.011</td>
  <td>52.58</td>
  <td></td>
</tr>
<tr>
  <td><a href="#eurovoc"><strong>Eurovoc</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.272</td>
  <td>1.523</td>
  <td>2.571</td>
  <td>9.468</td>
  <td></td>
</tr>
<tr>
  <td><a href="#eurovoc"><strong>Eurovoc</strong></a></td>
  <td><strong>it</strong></td>
  <td>0.245</td>
  <td>0.731</td>
  <td>1.527</td>
  <td>4.867</td>
  <td></td>
</tr>
<tr>
  <td><a href="#eurovoc"><strong>Eurovoc</strong></a></td>
  <td><strong>de</strong></td>
  <td>0.247</td>
  <td>0.678</td>
  <td>1.497</td>
  <td>4.915</td>
  <td></td>
</tr>
<tr>
  <td><a href="#eurovoc"><strong>Eurovoc</strong></a></td>
  <td><strong>es</strong></td>
  <td>0.246</td>
  <td>0.757</td>
  <td>1.411</td>
  <td>4.684</td>
  <td></td>
</tr>
<tr>
  <td><a href="#opendata"><strong>OpenData</strong></a></td>
  <td><strong>fr</strong></td>
  <td>1.169</td>
  <td>0.755</td>
  <td>1.209</td>
  <td>4.638</td>
  <td></td>
</tr>
<tr>
  <td><a href="#questionsecritesparlement"><strong>QuestionsEcritesParlement</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.189</td>
  <td>0.108</td>
  <td>0.156</td>
  <td>0.705</td>
  <td></td>
</tr>
<tr>
  <td><a href="#legi"><strong>LEGI</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.621</td>
  <td>0.088</td>
  <td>0.145</td>
  <td>0.563</td>
  <td></td>
</tr>
<tr>
  <td><a href="#amendementsparlement"><strong>AmendementsParlement</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.673</td>
  <td>0.045</td>
  <td>0.074</td>
  <td>0.274</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=wiki>Wiki</h4></td></tr>
<tr>
  <td><a href="#wikipedia-wikisource-wiktionary"><strong>Wikipedia</strong></a></td>
  <td><strong>en</strong></td>
  <td>6.893</td>
  <td>4.708</td>
  <td>7.898</td>
  <td>26.616</td>
  <td></td>
</tr>
<tr>
  <td><a href="#wikipedia-wikisource-wiktionary"><strong>Wikipedia</strong></a></td>
  <td><strong>de</strong></td>
  <td>2.877</td>
  <td>1.709</td>
  <td>3.476</td>
  <td>11.252</td>
  <td></td>
</tr>
<tr>
  <td><a href="#wikipedia-wikisource-wiktionary"><strong>Wikipedia</strong></a></td>
  <td><strong>fr</strong></td>
  <td>2.648</td>
  <td>1.726</td>
  <td>2.94</td>
  <td>9.879</td>
  <td></td>
</tr>
<tr>
  <td><a href="#wikipedia-wikisource-wiktionary"><strong>Wikipedia</strong></a></td>
  <td><strong>es</strong></td>
  <td>1.947</td>
  <td>1.245</td>
  <td>2.124</td>
  <td>7.161</td>
  <td></td>
</tr>
<tr>
  <td><a href="#wikipedia-wikisource-wiktionary"><strong>Wikipedia</strong></a></td>
  <td><strong>it</strong></td>
  <td>1.87</td>
  <td>1.06</td>
  <td>1.959</td>
  <td>6.161</td>
  <td></td>
</tr>
<tr>
  <td><a href="#wikisource"><strong>wikisource</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.186</td>
  <td>0.523</td>
  <td>0.795</td>
  <td>3.08</td>
  <td></td>
</tr>
<tr>
  <td><a href="#wiktionary"><strong>wiktionary</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.65</td>
  <td>0.053</td>
  <td>0.117</td>
  <td>0.347</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=math>Math</h4></td></tr>
<tr>
  <td><a href="#mathpile"><strong>MathPile</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.737</td>
  <td>3.408</td>
  <td>9.637</td>
  <td>27.29</td>
  <td></td>
</tr>
<tr>
  <td><a href="#pile-uncopyrighted"><strong>Pile (DM_Mathematics)</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.992</td>
  <td>1.746</td>
  <td>4.928</td>
  <td>8.127</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=forum>Forum</h4></td></tr>
<tr>
  <td><a href="#pile-uncopyrighted"><strong>Pile (StackExchange)</strong></a></td>
  <td><strong>en</strong></td>
  <td>15.269</td>
  <td>4.534</td>
  <td>10.275</td>
  <td>33.609</td>
  <td></td>
</tr>
<tr>
  <td><a href="#pile-uncopyrighted"><strong>Pile (Ubuntu_IRC)</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.01</td>
  <td>0.867</td>
  <td>2.159</td>
  <td>5.61</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=dialogue>Dialogue</h4></td></tr>
<tr>
  <td><a href="#claire-french-and-english"><strong>Claire</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.949</td>
  <td>0.818</td>
  <td>1.161</td>
  <td>4.709</td>
  <td><strong>DialogStudio</strong> (0.061 B words), <strong>BNC</strong> (0.011 B words), <strong>OANC</strong> (0.005 B words), <strong>AMI</strong> (0.001 B words), <strong>DailyDialog</strong> (0.001 B words), <strong>ICSI</strong> (0.001 B words)</td>
</tr>
<tr>
  <td><a href="#claire-french-and-english"><strong>Claire</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.037</td>
  <td>0.209</td>
  <td>0.31</td>
  <td>1.313</td>
  <td><strong>Senat</strong> (0.051 B words), <strong>Theatre</strong> (0.017 B words), <strong>ESLO</strong> (0.005 B words), <strong>CFPP</strong> (0.001 B words), <strong>OFROM</strong> (0.001 B words), <strong>ORFEO</strong> (0.001 B words), <strong>PFC</strong> (0.001 B words), <strong>SUMM</strong> (0.001 B words), <strong>TCOF</strong> (0.001 B words), <strong>ACSYNT</strong>, <strong>CID</strong>, <strong>CLAPI</strong>, <strong>FREDSum</strong>, <strong>LINAGORA</strong>, <strong>OTG</strong>, <strong>ParisStories</strong>, <strong>Rhapsodie</strong>, <strong>UBS</strong></td>
</tr>
<tr>
  <td><a href="#youtube"><strong>YouTube</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.038</td>
  <td>0.145</td>
  <td>0.336</td>
  <td>1.003</td>
  <td></td>
</tr>
<tr>
  <td><a href="#stac"><strong>Stac</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=legislative-transcripts>Legislative Transcripts</h4></td></tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>Europarl</strong></a></td>
  <td><strong>es</strong></td>
  <td>0.01</td>
  <td>0.052</td>
  <td>0.073</td>
  <td>0.325</td>
  <td></td>
</tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>Europarl</strong></a></td>
  <td><strong>de</strong></td>
  <td>0.01</td>
  <td>0.045</td>
  <td>0.073</td>
  <td>0.327</td>
  <td></td>
</tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>Europarl</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.01</td>
  <td>0.053</td>
  <td>0.072</td>
  <td>0.339</td>
  <td></td>
</tr>
<tr>
  <td><a href="#europarl-monolingual-and-parallel"><strong>Europarl</strong></a></td>
  <td><strong>en</strong></td>
  <td>0.011</td>
  <td>0.056</td>
  <td>0.069</td>
  <td>0.339</td>
  <td></td>
</tr>
<tr>
  <td><a href="#discourspublics"><strong>DiscoursPublics</strong></a></td>
  <td><strong>fr</strong></td>
  <td>0.11</td>
  <td>0.163</td>
  <td>0.238</td>
  <td>1.025</td>
  <td></td>
</tr>
<tr>
  <td><a href="#interventionsparlement"><strong>InterventionsParlement</strong></a></td>
  <td><strong>fr</strong></td>
  <td>1.832</td>
  <td>0.104</td>
  <td>0.157</td>
  <td>0.654</td>
  <td></td>
</tr>
<tr>
  <td colspan="7"><h4 id=programming>Programming</h4></td></tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>JAVASCRIPT</strong></td>
  <td>21.109</td>
  <td>8.526</td>
  <td>58.609</td>
  <td>141.647</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>JAVA</strong></td>
  <td>20.152</td>
  <td>7.421</td>
  <td>27.68</td>
  <td>89.297</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>C</strong></td>
  <td>8.626</td>
  <td>5.916</td>
  <td>24.092</td>
  <td>57.428</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>PHP</strong></td>
  <td>15.905</td>
  <td>4.865</td>
  <td>22.883</td>
  <td>66.844</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>PYTHON</strong></td>
  <td>12.962</td>
  <td>5.434</td>
  <td>21.683</td>
  <td>64.304</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>C++</strong></td>
  <td>6.378</td>
  <td>4.584</td>
  <td>18.835</td>
  <td>50.892</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>C#</strong></td>
  <td>10.839</td>
  <td>3.574</td>
  <td>13.381</td>
  <td>46.286</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>GO</strong></td>
  <td>4.73</td>
  <td>2.735</td>
  <td>10.262</td>
  <td>25.738</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>TYPESCRIPT</strong></td>
  <td>10.637</td>
  <td>2.617</td>
  <td>9.836</td>
  <td>28.815</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>RUST</strong></td>
  <td>1.387</td>
  <td>0.872</td>
  <td>3.241</td>
  <td>9.529</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>RUBY</strong></td>
  <td>3.405</td>
  <td>0.646</td>
  <td>2.392</td>
  <td>7.139</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>SWIFT</strong></td>
  <td>1.756</td>
  <td>0.553</td>
  <td>1.876</td>
  <td>6.134</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>KOTLIN</strong></td>
  <td>2.243</td>
  <td>0.454</td>
  <td>1.758</td>
  <td>5.769</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>SCALA</strong></td>
  <td>1.362</td>
  <td>0.457</td>
  <td>1.587</td>
  <td>4.862</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>TEX</strong></td>
  <td>0.398</td>
  <td>0.394</td>
  <td>1.507</td>
  <td>3.805</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>LUA</strong></td>
  <td>0.559</td>
  <td>0.318</td>
  <td>1.367</td>
  <td>3.279</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>DART</strong></td>
  <td>0.933</td>
  <td>0.308</td>
  <td>1.242</td>
  <td>3.864</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>PERL</strong></td>
  <td>0.392</td>
  <td>0.297</td>
  <td>1.149</td>
  <td>2.634</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>MATHEMATICA</strong></td>
  <td>0.027</td>
  <td>0.12</td>
  <td>1.117</td>
  <td>1.72</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>ASSEMBLY</strong></td>
  <td>0.248</td>
  <td>0.209</td>
  <td>0.867</td>
  <td>1.575</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>HASKELL</strong></td>
  <td>0.545</td>
  <td>0.307</td>
  <td>0.807</td>
  <td>2.364</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>FORTRAN</strong></td>
  <td>0.165</td>
  <td>0.192</td>
  <td>0.78</td>
  <td>1.843</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>JULIA</strong></td>
  <td>0.299</td>
  <td>0.152</td>
  <td>0.66</td>
  <td>1.539</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>OCAML</strong></td>
  <td>0.16</td>
  <td>0.13</td>
  <td>0.43</td>
  <td>1.107</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>ERLANG</strong></td>
  <td>0.099</td>
  <td>0.066</td>
  <td>0.26</td>
  <td>0.726</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>ELIXIR</strong></td>
  <td>0.282</td>
  <td>0.073</td>
  <td>0.258</td>
  <td>0.737</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>CLOJURE</strong></td>
  <td>0.126</td>
  <td>0.045</td>
  <td>0.179</td>
  <td>0.492</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>R</strong></td>
  <td>0.039</td>
  <td>0.028</td>
  <td>0.158</td>
  <td>0.305</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>MATLAB</strong></td>
  <td>0.001</td>
  <td>0.009</td>
  <td>0.043</td>
  <td>0.037</td>
  <td></td>
</tr>
<tr>
  <td><a href="#thestack"><strong>TheStack</strong></a></td>
  <td><strong>RACKET</strong></td>
  <td>0.004</td>
  <td>0.005</td>
  <td>0.015</td>
  <td>0.038</td>
  <td></td>
</tr>
</tbody>
</table>



### Details on Data Sources

#### AmendementsParlement
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4) ([nodeputes.fr](http://www.nosdeputes.fr/), [nossenateurs.fr](http://www.nossenateurs.fr/)). [API](https://github.com/regardscitoyens). License: [CC BY-SA](https://www.regardscitoyens.org/#&panel1-2).
* <u>Description</u>: A collection of proposed amendments by the French parliament: the legal text and description of the requested modification. 
* <u>Citation</u>: No paper found.

#### AmericanStories
* <u>Source</u>: [dell-research-harvard/AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories). License: [CC BY 4.0](https://huggingface.co/datasets/dell-research-harvard/AmericanStories).
* <u>Extracted from</u>: [Chronicling America](https://www.loc.gov/collections/chronicling-america/about-this-collection/). License: [Open](https://www.loc.gov/collections/chronicling-america/about-this-collection/rights-and-access/).
* <u>Description</u>: "The American Stories dataset is a collection of full article texts extracted from historical U.S. newspaper images. It includes nearly 20 million scans from the public domain Chronicling America collection maintained by the Library of Congress. The dataset is designed to address the challenges posed by complex layouts and low OCR quality in existing newspaper datasets" (from the [dataset card](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)).
* <u>Citation</u>: Melissa Dell, Jacob Carlson, Tom Bryan, Emily Silcock, Abhishek Arora, Zejiang Shen, Luca D'Amico-Wong, Quan Le, Pablo Querubin and Leander Heldring (2023). "American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers," [arxiv:2308.12477](https://arxiv.org/abs/2308.12477v1).


#### Claire (French and English)
* <u>Sources</u>:
  * French dataset: [OpenLLM-France/Claire-Dialogue-French-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1).
  * English dataset: [OpenLLM-France/Claire-Dialogue-English-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1).
* <u>Description</u>: The Claire datasets are composed of transcripts of spoken conversation -- including parliamentary proceedings, interviews, debates, meetings, and free conversations -- as well as some written conversations from theater plays and written chats. The dataset is designed to help downstream performance of models fine-tuned for tasks requiring the comprehension of spontaneous spoken conversation, such as meeting summarization. Each dialogue is split into speech turns, and each speech turn is labeled with the name of the speaker or a unique identifier.
* <u>Citation</u>: Julie Hunter, Jérôme Louradour, Virgile Rennard, Ismaïl Harrando, Guokan Shang, Jean-Pierre Lorré (2023). The Claire French Dialogue Dataset. [arXiv:2311.16840](https://arxiv.org/abs/2311.16840).

#### CroissantAligned
* <u>Source</u>: [croissantllm/croissant_dataset_no_web_data](https://huggingface.co/datasets/croissantllm/croissant_dataset_no_web_data). License: not specified.
* <u>Extracted from</u>: [OPUS](https://opus.nlpl.eu/), theses, [song lyrics](https://www.lacoccinelle.net)
* <u>Description</u>: A collection of English-French translation pairs selected by a custom filtering pipeline. Designed to "improve the multilingual capabilities of the model" ([Arxiv paper](https://arxiv.org/pdf/2402.00786)).
* <u>Citation</u>: Manuel Faysse, Patrick Fernandes, Nuno M. Guerreiro, António Loison, Duarte M. Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro H. Martins, Antoni Bigata Casademunt, François Yvon, André F.T. Martins, Gautier Viaud, Céline Hudelot, Pierre Colombo (2024). "CroissantLLM: A Truly Bilingual French-English Language Model," [arXiv:2402.00786](https://arxiv.org/abs/2402.00786).

#### DiscoursPublics
  * <u>Source</u>: Corpus contributed by OpenLLM partners.
  * <u>Extracted from</u>: [Vie Publique](https://www.vie-publique.fr/collection-discours-publics).
  * <u>Description</u>: A collection of public speeches from the principal public actors in France including speeches from the French President starting from 1974 and from the Prime Minister and members of the government starting from 1980.
  * <u>Citation</u>: No paper found.

#### Europarl (monolingual and parallel)
* <u>Sources</u>: 
  * `fr-en`, `es-en`, `it-en` parallel data: [Europarl v7](https://www.statmt.org/europarl/v7/). License: [Open](https://www.statmt.org/europarl/).
  * `fr`, `en`, `de`, `es` monolingual data and `de-fr` parallel data: [Europarl v10](https://www.statmt.org/europarl/v10/training-monolingual/). License: [Open](https://www.statmt.org/europarl/).
* <u>Description</u>: "The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek. The goal of the extraction and processing was to generate sentence aligned text for statistical machine translation systems" ([www.statmt.org](https://www.statmt.org/europarl/)).
* <u>Citation</u>: Philipp Koehn (2005). "Europarl: A Parallel Corpus for Statistical Machine Translation," MT Summit. 

#### Eurovoc
* <u>Source</u>:   [EuropeanParliament/Eurovoc](https://huggingface.co/datasets/EuropeanParliament/Eurovoc). License: [EUPL 1.1](https://joinup.ec.europa.eu/licence/european-union-public-licence-version-11-or-later-eupl).
* <u>Extracted from</u>: [Cellar](https://op.europa.eu/en/web/cellar). License: [Open](https://op.europa.eu/en/web/cellar).
* <u>Description</u>: A collection of mutlilingual documents from the data repository of the Publications Office of the European Union annotated with Eurovoc labels. 
* <u>Citations</u>:
  * Ilias Chalkidis, Emmanouil Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos (2019). "Extreme Multi-Label Legal Text Classification: A Case Study in EU Legislation," Proceedings of the Natural Legal Language Processing Workshop 2019, pages 78–87, Minneapolis, Minnesota. Association for Computational Linguistics.
  * Ilias Chalkidis,  Manos Fergadiotis, Prodromos Malakasiotis and Ion Androutsopoulos (2019). "Large-Scale Multi-Label Text Classification on EU Legislation," Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers).
  * Andrei-Marius Avram, Vasile Pais, and Dan Ioan Tufis (2021). "PyEuroVoc: A Tool for Multilingual Legal Document Classification with EuroVoc Descriptors," Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 92–101, Held Online. INCOMA Ltd.
  * Zein Shaheen, Gerhard Wohlgenannt and Erwin Filtz (2020). "Large scale legal text classification using transformer models," [arXiv:2010.12871](https://arxiv.org/abs/2010.12871v1).

#### FineWebEdu
* <u>Source</u>: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
* <u>Extracted from</u>: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
* <u>Description</u>: A 1.3 trillion token selection from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), which contains 15 trillion tokens of curated data from 96 Common Crawl dumps. Content in FineWebEdu has been selected by a custom designed classifier for its high-quality, educational content. 
* <u>Citation</u>: Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," [	arXiv:2406.17557](https://arxiv.org/abs/2406.17557).

#### GallicaMonographies
* <u>Source</u>: Corpus contributed by OpenLLM partners. A version is also published here: [PleIAs/French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books). License: None (public domain).
* <u>Extracted from</u>: [Gallicagram](https://shiny.ens-paris-saclay.fr/app/gallicagram).
* <u>Description</u>: A large collection of French monographies in the public domain made available through the French National Library ([Gallica](https://gallica.bnf.fr/accueil/fr/content/accueil-fr?mode=desktop)).
* <u>Citation</u>: No paper found.

#### GallicaPress
* <u>Source</u>: Corpus contributed by OpenLLM partners. A version is also published here: [PleIAs/French-PD-Newspapers](https://huggingface.co/datasets/PleIAs/French-PD-Newspapers). License: None (public domain).
* <u>Extracted from</u>: [Gallicagram](https://shiny.ens-paris-saclay.fr/app/gallicagram).
* <u>Description</u>: A large collection of French newspapers and periodicals in the public domain made available through the French National Library ([Gallica](https://gallica.bnf.fr/accueil/fr/content/accueil-fr?mode=desktop)).
* <u>Citation</u>: No paper found.

#### Gutenberg
  * <u>Source</u>: Corpus compiled by OpenLLM partners.
  * <u>Extracted from</u>: 
    * [aleph.gutenberg.org](http://aleph.gutenberg.org/) via [Project Gutenberg](https://www.gutenberg.org/). License: [Open](https://www.gutenberg.org/policy/terms_of_use.html).
    * [pgcorpus](https://github.com/pgcorpus/gutenberg). License: [CC BY-4.0](https://zenodo.org/records/2422561).
  * <u>Description</u>: A collection of free eBooks, manually prepared by human annotators. 
  * <u>Citation</u>: No paper found.

#### HAL
* <u>Source</u>:
* <u>Extracted from</u>: [HAL](https://hal.science/).
* <u>Description</u>: A collection of scientific papers and manuscripts distributed through an open science platform.
* <u>Citation</u>: 

#### InterventionsParlement
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4) ([nodeputes.fr](http://www.nosdeputes.fr/), [nossenateurs.fr](http://www.nossenateurs.fr/)). [API](https://github.com/regardscitoyens). License: [CC BY-SA](https://www.regardscitoyens.org/#&panel1-2).
* <u>Description</u>: Transcripts of speeches made during French parlementary debates.  
* <u>Citation</u>: No paper found.

#### MathPile
* <u>Source</u>: [GAIR/MathPile_Commercial](https://huggingface.co/datasets/GAIR/MathPile_Commercial). License: CC BY-SA 4.0
* <u>Extracted from</u>: [MathPile](https://huggingface.co/datasets/GAIR/MathPile). License: CC BY-SA-NC 4.0.
* <u>Description</u>: A preprocessed collection of documents focused on math, including Textbooks, arXiv, Wikipedia, ProofWiki, StackExchange, and web pages from Common Crawl. The content targets a range of levels, from kindergarten through postgraduate level. MathPile_Commercial was obtained by removing documents from MathPile that do not allow commercial use.
* <u>Citation</u>: Zengzhi Wang, Rui Xia and Pengfei Liu (2023). "Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math," [	arXiv:2312.17120](https://export.arxiv.org/abs/2312.17120).

#### OpenData
* <u>Source</u>: [Nicolas-BZRD/DILA_OPENDATA_FR_2023](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main) (balo, dole, inca, kali, legi and sarde subsets). License: ODC-BY.
* <u>Extracted from</u>: [OpenData](https://echanges.dila.gouv.fr/OPENDATA/) (Data collection date: October, 2023).
* <u>Description</u>: "The French Government Open Data (DILA) Dataset is a collection of text data extracted from various sources provided by the French government, specifically the Direction de l'information légale et administrative (DILA). This dataset contains a wide range of legal, administrative, and legislative documents. The data has been organized into several categories for easy access and analysis" (from the [dataset card](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main)).
* <u>Citation</u>: No paper found.

#### OpenEdition
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>: [Open Edition](https://www.openedition.org/).
* <u>Description</u>: 
* <u>Citation</u>: No paper found.

#### PeS2o
* <u>Source</u>: [allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o). License: 	[ODC BY-v1.0](https://opendatacommons.org/licenses/by/1-0/)
* <u>Extracted from</u>: [S2ORC](https://github.com/allenai/s2orc) (see [aclanthology](https://aclanthology.org/2020.acl-main.447/)). Knowledge cutoff: 2023-01-03.
* <u>Description</u>: A preprocessed collection of academic papers designed for pre-training of language models. It includes a subset of full papers and another subset of titles and abstracts.
* <u>Citation</u>: Luca Soldaini and Kyle Lo (2023). "peS2o (Pretraining Efficiently on S2ORC) Dataset}, Allen Institute for AI. [GitHub](https://github.com/allenai/pes2o).

#### Pile (Uncopyrighted)
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

#### QuestionsEcritesParlement
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4) ([text](https://data.regardscitoyens.org/nosdeputes.fr/)). License: [CC BY-NC-SA](https://data.regardscitoyens.org/nosdeputes.fr/).
* <u>Description</u>: Collection of long written questions, read during a session at the french national assembly: from a member of french parliament to a minister  (Minister got 2 month to respond). ([text](https://data.regardscitoyens.org/nosdeputes.fr/)).
* <u>Citation</u>: No paper found.

#### RedPajama (v2)
* <u>Source</u>: [togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2). License: Apache 2.0 (data preparation code), Not specified (data) but see [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).
* <u>Description</u>: "RedPajama-V2 is an open dataset for training large language models. The dataset includes over 100B text documents coming from 84 CommonCrawl snapshots and processed using the [CCNet](https://github.com/facebookresearch/cc_net) pipeline. Out of these, there are 30B documents in the corpus that additionally come with quality signals, and 20B documents that are deduplicated" (from [GitHub](https://github.com/togethercomputer/RedPajama-Data)).
* <u>Citation</u>: Together Computer (2023). "RedPajama-Data-v2: an Open Dataset with 30 Trillion Tokens for Training Large Language Models," [GitHub](https://github.com/togethercomputer/RedPajama-Data).

#### STAC
* <u>Source</u>: [STAC](https://www.irit.fr/STAC/corpus.html). License: CC BY-SA-NC 4.0.
* <u>Description</u>: A collection of chats from an online version of the game Settlers of Catan.
* <u>Citation</u>: Nicholas Asher, Julie Hunter, Mathieu Morey, Farah Benamara and Stergos Afantenos (2016). "Discourse structure and dialogue acts in multiparty dialogue: the STAC corpus," The Tenth International Conference on Language Resources and Evaluation (LREC 2016). European Language Resources Association, pp. 2721-2727.

#### TheStack
* <u>Source</u>: [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup). License: Other (mixture of copyleft licenses).
* <u>Description</u>: "The Stack contains over 6TB of permissively-licensed source code files covering 358 programming languages. The dataset was created as part of the BigCode Project, an open scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs). The Stack serves as a pre-training dataset for Code LLMs, i.e., code-generating AI systems which enable the synthesis of programs from natural language descriptions as well as other from code snippets. This is the near-deduplicated version with 3TB data" (from the [dataset card](https://huggingface.co/datasets/bigcode/the-stack-dedup)).
* <u>Citation</u>: Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra and Harm de Vries (2022). "The Stack: 3 TB of permissively licensed source code," [arxiv:2211.15533](https://arxiv.org/abs/2211.15533).

#### Theses
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>: [theses.fr](https://theses.fr/?domaine=theses) and  [HAL???]().
* <u>Description</u>: 
* <u>Citation</u>: No paper found.

#### Wikipedia, Wikisource, Wiktionary
* <u>Source</u>: Corpus contributed by LINAGORA Labs (OpenLLM-France).
  Also published here:
  * [OpenLLM-France/wikipedia](https://huggingface.co/datasets/OpenLLM-France/wikipedia)
  * [OpenLLM-France/wikisource](https://huggingface.co/datasets/OpenLLM-France/wikisource)
  * [OpenLLM-France/wiktionary](https://huggingface.co/datasets/OpenLLM-France/wiktionary)
* <u>Extracted from</u>: [Wikimedia dumps](https://dumps.wikimedia.org/other/enterprise_html/runs/). License: [GFDL/CC BY-SA](https://dumps.wikimedia.org/legal.html)
* <u>Description</u>:
* <u>Citation</u>: No paper found.

#### YouTube
* <u>Source</u>: Corpus contributed by LINAGORA Labs (OpenLLM-France).
* <u>Extracted from</u>:
* <u>Description</u>: 
* <u>Citation</u>: No paper found.

## Example use in python

Load the dataset using the `datasets` library:
```python
from datasets import load_dataset

kwargs = {"split": "train", "streaming": True}

dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", **kwargs)

for sample in dataset:
   text = sample["text"]
   # ... do something with the text
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
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "code-python", **kwargs)
```
Load data from Wikipedia (in available languages):
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Wikipedia", **kwargs)
```
Load data from French pages of Wikipedia ([wikipedia.fr](https://www.wikipedia.fr/)):
```python
dataset = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Wikipedia-fr", **kwargs)
```

## License

TODO

## Citation

TODO

## Contact

<pre>contact@openllm-france.fr</pre>