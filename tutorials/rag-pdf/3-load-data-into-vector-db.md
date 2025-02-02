# 3 - Load Data into a Vector Database

At this stage, we are going to save the processed PDF data into a vector database.

code: [rag_2_load_data_into_milvus.ipynb](../../examples/notebooks/rag-pdf-1/rag_2_load_data_into_milvus.ipynb)


<img src="media/rag-overview-2b.png" style="max-width:90%;"/>



##  Milvus - An Open Source Vector Database

Our vector database of chioce is **Milvus**

Milvus is an open-source vector database designed to manage and query large-scale vector data. Built to handle unstructured data like embeddings from text, images, and audio, Milvus is highly optimized for similarity search and nearest neighbor search. 

Read more about Milvus at [milvus.io](https://milvus.io/)

## Step-1: Loading Configuration

```python
from my_config import MY_CONFIG
```

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

We are also renaming the dataframes columns as follows, so we confirm to Milvus client library convention.

- `embeddings` --> `vector`
- `contents` --> `text`

At the end of this step, out data will look like this:

<img src="media/loading-to-milvus-1.png" style="max-width:90%;"/>

## Step-3: Connecting to Milvus Database

In this step we  are connecting to an embedded Milvus instance - it is a file named `rag_1_dpk.db`.  For examples, embedded databases are fine.  In production we would connect to an instance running in the cloud.

```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(MY_CONFIG.DB_URI)
```

## Step-4: Creating a Milvus Collection

First we clear the collection if it exists -- so the data import is clean.

Then we are creating a collection with these properties:

- **`collection_name = 'dpk_papers'`** : name of collection
- **`dimension=384`**: This should match the embedding model's (that we used to create embeddings for chunks) output length.
- **`metric_type='IP'`**: This the metric to be used when finding similar items.  IP = Inner product distance
- **`consistency_level='Strong'`**: Milvus supports various consistency levels.  For more details, check [Milvus documentation](https://milvus.io/docs/consistency.md)
- **`auto_id=True`** : Milvus will assign primary ids as we insert data.

Consult [Milvus collection documentation](https://milvus.io/docs/create-collection.md) for a deep dive into creating collections.


Here is the (abbreviated) code.


```python
# if we already have a collection, clear it first
if milvus_client.has_collection(collection_name='dpk_papers'):
    milvus_client.drop_collection(collection_name='dpk_papers')


milvus_client.create_collection(
    collection_name='dpk_papers',
    dimension=384,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
    auto_id=True
)
```

Then we insert the the data into the collection.

```python
res = milvus_client.insert(collection_name='dpk_papers', 
                            data=data_df.to_dict('records'))

print('inserted # rows', res['insert_count'])
milvus_client.get_collection_stats('dpk_papers')
```

Output is 

```text
inserted # rows 60
```

And finally we close the Milvus connection

```python
milvus_client.close()
```

## Conclusion

In this step, we have loaded processed chunks into Milvus database.

Next step is [performing vector search on our data](4-vector-search.md)