# 4 - Perform Vector Search

Now that we’ve successfully imported our data and embeddings into the **Milvus vector database**, it’s time to run some powerful queries!

Unlike traditional **keyword searches**, these queries perform **semantic searches**, meaning they retrieve results based on meaning rather than exact word matches.


💻 **Code Reference**: [rag_3_vector_search.ipynb](../../examples/notebooks/rag-pdf-1/rag_3_vector_search.ipynb)


<img src="media/rag-overview-2c.png" style="max-width:90%;"/>


## Step-1: Load Configuration

First, let’s load our configuration settings:


```python
from my_config import MY_CONFIG
```

---

## Step-2: Connect to Vector Database

We’ll be connecting to an **embedded Milvus instance**, which is a local database file (`rag_1_dpk.db`). While this setup is great for testing, a production environment would typically connect to a cloud-hosted Milvus instance.

```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(MY_CONFIG.DB_URI)
```

---

## Step-3: Setup Embeddings

Before we can search, we need to convert our query strings into vector embeddings using the same embedding model we used earlier.

Here is the code:


```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(MY_CONFIG.EMBEDDING_MODEL)

def get_embeddings (str):
    embeddings = embedding_model.encode(str, normalize_embeddings=True)
    return embeddings
```

### **Code Breakdown:**

- We set the **`HF_ENDPOINT`** to Hugging Face’s mirror to ensure smooth model downloads.
- The **`get_embeddings`** function converts a text query into an embedding vector.

### **Testing the Embedding Conversion:**


```python
# Test Embeddings
text = 'Paris 2024 Olympics'
embeddings = get_embeddings(text)
print ('sentence transformer : embeddings len =', len(embeddings))
print ('sentence transformer : embeddings[:5] = ', embeddings[:5])
```

**Sample Output:**

```text
sentence transformer : embeddings len = 384
sentence transformer : embeddings[:5] =  [ 0.02468892  0.10352131  0.0275264  -0.08551715 -0.01412829]
```

## Step-4: Perform Vector Search

### 4.1 - Search Helper functions

To streamline our search, we define a helper function:

```python
def  do_vector_search (query):
    query_vectors = [get_embeddings(query)] 

    results = milvus_client.search(
        collection_name=MY_CONFIG.COLLECTION_NAME,  # target collection
        data=query_vectors,  # query vectors
        limit=5,  # number of returned entities
        output_fields=["filename", "page_number", "text"],  
        # specifies fields to be returned
    )
    return results
## ----
```

### **Code Breakdown:**

- Converts the query text into an embedding vector.
- Executes a **vector search** in the Milvus database.
- Returns the **top 5** most relevant results.

### **4.2 - Running a Search Query**

Now, let’s test it out!

```python
query = "What was the training data used to train Granite models?"

results = do_vector_search (query)
print_search_results(results)
```

This will output the top matches, ranked by **search score** (higher scores indicate better matches, with a range of **0 to 1.0**).

**Sample Output:**


```text
num results :  5
------ result 1 --------
search score: 0.5530709028244019
filename: granite.pdf
text: ...


 ------ result 2 --------
search score: 0.477556437253952
filename: granite.pdf
text: ...

...
```

From this output, we can see that the search has successfully retrieved relevant results, with the top match scoring **0.553**.

## Conclusion

In this guide, we’ve demonstrated how to perform a **semantic vector search** using Milvus. By leveraging embeddings, we’ve moved beyond simple keyword searches to retrieving results based on meaning—enhancing the accuracy and relevance of search results.

Next step is **[performing RAG queries using an LLM](5-query-LLM.md)**