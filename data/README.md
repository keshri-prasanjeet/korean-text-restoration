# Data

`processed/` holds the train, validation and test splits used in this project. They are
built by `scripts/build_dataset.py` from Korean Wikipedia (Hugging Face `wikimedia/wikipedia`,
config `20231101.ko`, file `train-00002-of-00003.parquet`).

Korean Wikipedia text is licensed under CC BY-SA 4.0. These files contain sentences from it,
so they are distributed under the same license. Attribution: Wikipedia contributors, Korean
Wikipedia, dump of 1 November 2023. Each row keeps the source `article_id`, so a sentence can
be traced back to its article (`https://ko.wikipedia.org/?curid=<article_id>`).

`raw/` is not tracked by git; download the parquet file into it to rebuild the splits.
