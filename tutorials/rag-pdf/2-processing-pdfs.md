# 2 - Processing PDFs

In this stage we will process PDF documents, getting them ready for queries.

The steps are:

- Read PDF documents, extract contents of PDFs
- remove duplicates
- split them into chunks
- create embeddings for chunks

Code: [examples/notebooks/rag/rag_1A_dpk_process_python.ipynb](../../examples/notebooks/rag/rag_1A_dpk_process_python.ipynb)

## Step-1: Configuration

We have a common configuration parameters defined in [my_config.py](../../examples/notebooks/rag/my_config.py)

Here is a snippet of `my_config.py`

```python

class MyConfig:
    pass 

MY_CONFIG = MyConfig ()

## Input Data - configure this to the folder we want to process
MY_CONFIG.INPUT_DATA_DIR = "input"
MY_CONFIG.OUTPUT_FOLDER = "output"
MY_CONFIG.OUTPUT_FOLDER_FINAL = os.path.join(MY_CONFIG.OUTPUT_FOLDER , "output_final")
### -------------------------------

### Milvus config
MY_CONFIG.DB_URI = './rag_1_dpk.db'  # For embedded instance
MY_CONFIG.COLLECTION_NAME = 'dpk_papers'


## Embedding model
MY_CONFIG.EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MY_CONFIG.EMBEDDING_LENGTH = 384

## LLM Model
MY_CONFIG.LLM_MODEL = "ibm-granite/granite-3.0-8b-instruct"
```

Here we are defining a  place holder class `MyConfig` that contains all our config parameters.  This is a good practice, so all our configuration parameters are confined to one class (and not polluting global namespace)

You can add a new configuration parameter to this class very easily:

```python
MY_CONFIG.SAMPLE_CONFIG = "hello"
```

And in our code we initialize this configuration as follows

```python
from my_config import MY_CONFIG
```


## Step-2: Downloading Data

Here we are downloading two PDF documents from [archive.org](https://arxiv.org).

We are using a util function `download_file` defined in [utils.py](../../eamples/notebooks/rag/utils.py)

`download_file` checks if the local file exists, and downloads it if it doesn't

```python
from utils import download_file

download_file (url = 'https://arxiv.org/pdf/1706.03762', local_file = os.path.join(MY_CONFIG.INPUT_DATA_DIR, 'attention.pdf' ))
```

## Step 2.2 - Setting up Input Output Directories

We will process PDF documents in multiple stages.  

So for example during stage-1,  `pdf2parquet` process, reads data from `input` directory and produces output into  `output/01_parquet_out`.

`input ---(stage-1)--> output/01_stage1_out ---(stage 2)---> output/02_stage2_out` 

This output (`output/01_stage1_out`) becomes input to the next stage.  And so on.

Here are processing chain (TODO : update)

```
Input : input

output
├── 01_parquet_out
├── 02_chunk_out
├── 03_docid_out
├── 04_exact_dedupe_out
├── 05_embeddings_out
└── output_final
```

## 📝 Sidebar: Why Parquet format?

The Data Prep Kit utilizes the **parquet** format to temporarily store intermediate data generated during processing stages. This efficient storage solution offers numerous benefits, including:


* **Compressed storage**: Reduces storage requirements and saves space
* **Fast query performance**: Enables rapid analysis and insights
* And many more features that enhance overall data management

**Why Parquet Format is a Popular Choice**

Parquet format is widely adopted in Big Data applications due to its exceptional performance and efficiency. Its popularity can be attributed to its ability to handle large datasets with ease.

**Learn More About Parquet Format**

For a deeper understanding of parquet format, visit the following resources:

* [Apache Parquet Official Site](https://parquet.apache.org/)
* [Databricks: What is Parquet?](https://www.databricks.com/glossary/what-is-parquet)

## Step-3: Extracting Contents from PDF (PDF2Parquet)

Let's start by extracting the content from PDFs.



Here is the code that does it

```python
TODO: new API
```

Parameters explained:
- TODO




