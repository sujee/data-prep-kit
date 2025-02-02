# 3 - Load Data into a Vector Database

Now that we've processed our PDF data, the next step is to store it in a **vector database** for efficient retrieval. In this guide, we'll be using **Milvus**, a powerful open-source vector database designed for handling unstructured data like text embeddings. 

💻 **Code Reference**: [rag_2_load_data_into_milvus.ipynb](../../examples/notebooks/rag-pdf-1/rag_2_load_data_into_milvus.ipynb)


<img src="media/rag-overview-2b.png" style="max-width:90%;"/>

---

##  Why Milvus?

Milvus is built for high-performance **vector similarity search** and is optimized for working with embeddings generated from text, images, and audio. It allows for fast and scalable querying, making it ideal for **retrieval-augmented generation (RAG) workflows** like ours.  

🔗 Learn more about Milvus: [milvus.io](https://milvus.io/)

---

## Step-1: Load Configuration

Before we begin, let’s import our configuration settings:  


```python
from my_config import MY_CONFIG
```
---

## Step-2: Load Processed PDF data

We'll now load the data we saved in the previous step (**pdf2parquet**) from the `output/output_final` directory. This folder contains **Parquet** files—one for each input PDF. 

The following code does the following:

- Reads all Parquet files using `pandas.read_parquet`  
-  Combines them into a single DataFrame using `pandas.concat`  

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

We are also renaming the  columns as follows, to match Milvus’ expected format.

- `embeddings` --> `vector`
- `contents` --> `text`

```python
data_df = data_df.rename( columns= {'embeddings' : 'vector', 'contents' : 'text'})
```

**After this step, our processed data looks like this:** 

<img src="media/loading-to-milvus-1.png" style="max-width:90%;"/>

---

## Step-3: Connect to Milvus

We’re connecting to an **embedded Milvus instance** (a local database file `rag_1_dpk.db`). While this works for testing, in production, you’d connect to a cloud-hosted instance.  

```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(MY_CONFIG.DB_URI)
```

---

## Step-4: Create a Milvus Collection

Before inserting data, we **clear any existing collection** to ensure a fresh import.  

Then we are creating a collection with these properties:

- **`collection_name = 'dpk_papers'`** → name of our collection
- **`dimension=384`**: → Matches the output length of our embedding model  
- **`metric_type='IP'`** → Uses **Inner Product Distance** for similarity search
- **`consistency_level='Strong'`** → Ensures consistent query results.  For more details, check [Milvus documentation](https://milvus.io/docs/consistency.md)
- **`auto_id=True`** → Milvus assigns primary IDs automatically  

🔗 **More on Milvus collections**: [Milvus Docs](https://milvus.io/docs/create-collection.md) 


### **Code: Creating the Collection**  

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

---

## Step 5: Insert Data into Milvus

Now that our collection is set up, let’s **insert our processed data** into Milvus.  

```python
res = milvus_client.insert(collection_name='dpk_papers', 
                            data=data_df.to_dict('records'))

print('inserted # rows', res['insert_count'])
milvus_client.get_collection_stats('dpk_papers')
```

**Example Output:**  

```text
inserted # rows 60
```

---

## Step 6: Close the Connection

Once the data is successfully stored, we **close the Milvus connection** to free up resources.  

```python
milvus_client.close()
```

---

## Wrapping Up

In this guide, we:  
✔ Created a **vector collection** for similarity search  in Milvus  
✔ Loaded processed PDF data into **Milvus**  

🚀 **Next up:** Performing **vector search** to retrieve relevant content from our dataset!  