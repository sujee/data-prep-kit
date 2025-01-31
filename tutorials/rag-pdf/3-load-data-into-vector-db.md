# 3 - Load Data into a Vector Database

At this stage, we are going to save the processed PDF data into a vector database.

code : [rag_1B_load_data_into_milvus.ipynb](../../examples/notebooks/rag-pdf/rag_1B_load_data_into_milvus.ipynb)


##  Milvus - An Open Source Vector Database

Our vector database of chioce is **Milvus**

Milvus is an open-source vector database designed to manage and query large-scale vector data. Built to handle unstructured data like embeddings from text, images, and audio, Milvus is highly optimized for similarity search and nearest neighbor search. 

Read more about Milvus at [milvus.io/](https://milvus.io/)

## Step-2: Loading Processed PDF data

We will load the data saved from previous stage (pdf2parquet) from `output/output_final`

This directory will contain a bunch of parquet files - one for each input PDF.

The following code will load all pq files:

- It reads individual parquet files using pandas `pandas.read_parquet`
- Then it merges all dataframes into one (`data_df`) using `pandas.concat` method


```python
import pandas as pd
import glob

# Get a list of all Parquet files in the directory
parquet_files = glob.glob(f'{MY_CONFIG.OUTPUT_FOLDER_FINAL}/*.parquet')

# Create an empty list to store the DataFrames
dfs = []

# Loop through each Parquet file and read it into a DataFrame
for file in parquet_files:
    df = pd.read_parquet(file)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
data_df = pd.concat(dfs, ignore_index=True)
```

We are also renaming the dataframes columns as follows, just so we confirm to Milvus client library convention.

- `embeddings` --> `vector`
- `contents` --> `text`

At the end of this step, out data will look like this:

TODO: insert diagram or table

