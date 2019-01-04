# Improving Retriveal and Comprehension on HotpotQA

Research based on [(Yang, Qi, Zhang, et al. 2018)](https://hotpotqa.github.io/)
 
## Setup and Installation

First you need Indri. Installation instructions depending on 
operating system are available [here](https://github.com/nickvosk/pyndri).
 
Clone repository with `git`.
```git
git clone https://github.com/janaleible/hotpotQA-ir-task.git
```
Create a virtual environment with `python3.6`:
```bash
python -m venv venv
```
Activate virtual environment.
```bash
source venv/bin/activate
```
Install requirements with `pip`.
```bash
pip install 
```
If pyndri does not compile, follow the instruction [here](https://github.com/nickvosk/pyndri)
with the virtual environment active.

## Pre-process Data.
Get the raw wiki data from [HotpotQA](https://hotpotqa.github.io/wiki-readme.html). 
Uncompress the file in `./data/raw`.

From the root of the project, run
```bash
(1) python data_processing.py -a title
```
```bash
(2) python data_processing.py -a trec
```
Process (1) will build a mapping from document title to wikipedia ID
and vice-versa. The `title2wid` has the structure `Dict[str, List[int]]`
because some titles my not be disambiguated. Tests show this only happens
for one article with title: `Harry Diamond`. TODO: ignore one and create a
fully one-to-one mapping. 

Process (2) will build trec TEXT files to be indexed with Indri. If you
are running into memory issues, set the flag `--use_less_memory 1`. This
will use constant memory and will be faster in building the index. A side
effect is that the index will be slightly slower and that the original
will be much harder to recover from an external id.

## Build Index
Once you have the TREC files run indri from the project root.
```bash
IndriBuildIndex build_indri_index.xml
```

## Load the Index and Experiment
The module `retrieval.index` provides an interface to the index and 
mappings. Some obvious functionality is implemented but much more is
possible. Check out [pyndri](https://github.com/cvangysel/pyndri),
[IndriBuildIndex](https://sourceforge.net/p/lemur/wiki/IndriBuildIndex%20Parameters/), 
[IndriQueryLanguage](https://sourceforge.net/p/lemur/wiki/The%20Indri%20Query%20Language/),
and [IndriQueryLanguageReference](https://sourceforge.net/p/lemur/wiki/Indri%20Query%20Language%20Reference/)
for ideas.

A known issue is that there is no easy way to retrieve the original TREC
document from an external id. Right now you have to match the external
id in the range of ids provided by the file `.trectext` file names, 
parse the xml until you find the document id and retrieve the `<TEXT>` 
tag.





 
 
 
 
