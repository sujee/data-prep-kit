# Querying Documents using LLM

Welcome to the final step of our RAG (Retrieval-Augmented Generation) pipeline! Here, we’ll explore how to query documents effectively using a Large Language Model (LLM).

💻 **Code Reference**: [rag_4_query_replicate.ipynb](../../examples/notebooks/rag-pdf-1/rag_4_query_replicate.ipynb)


<img src="media/rag-overview-2d.png" style="max-width:90%;"/>

---

## Step-1: Load Configuration

First, let’s load our configuration settings:


```python
from my_config import MY_CONFIG
```

---

## Step-2: Load settings from `.env` file

We need to load the **`REPLICATE_API_TOKEN`** from our `.env` file to authenticate API calls to the **Replicate** service.

Code:

```python
from dotenv import find_dotenv, dotenv_values

config = dotenv_values(find_dotenv())

MY_CONFIG.REPLICATE_API_TOKEN = config.get('REPLICATE_API_TOKEN')
```

### 💡 **Why Use `.env` Files?**

`.env` files store sensitive configurations such as database passwords and API keys. These files should remain private and should not be checked into the codebase.

---

## Step-3: Connect to Vector Database

We'll establish a connection to an **embedded Milvus instance**, which stores our document embeddings. For production environments, a cloud-hosted Milvus instance is recommended.

```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(MY_CONFIG.DB_URI)
```

---

## Step-4: Setup Embeddings

An embedding model converts text into vector representations for efficient searching.


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

---

## Step-5: Vector search handy functions

Before querying the LLM, we need to retrieve relevant documents using vector search..

```python
def fetch_relevant_documents (query : str) :
    search_res = milvus_client.search(
        collection_name=MY_CONFIG.COLLECTION_NAME,
        data = [get_embeddings(query)], 
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    retrieved_docs_with_distances = [
        {'text': res["entity"]["text"], 'distance' : res["distance"]} for res in search_res[0]
    ]
    return retrieved_docs_with_distances
```

### **Code breakdown**

- **`milvus_client.search`** performs a **semantic search** using **Inner Product (IP) distance**.
- Retrieves **top 3** most relevant documents.

---

## Step-6: Initialize LLM

We will use **[replicate](https://replicate.com/)** service to run LLMs.  Replicate allows calling LLMs using easy to use APIs.

### **6.1 - Choosing an LLM**

We’ll use an open-source model from the available options.

We will proceed with **`ibm-granite/granite-3.0-8b-instruct`**.



| Model                               | Publisher | Params | Description                                          |
|-------------------------------------|-----------|--------|------------------------------------------------------|
| ibm-granite/granite-3.0-8b-instruct | IBM       | 8 B    | IBM's newest Granite Model v3.0  (default)           |
| ibm-granite/granite-3.0-2b-instruct | IBM       | 2 B    | IBM's newest Granite Model v3.0                      |
| meta/meta-llama-3.1-405b-instruct   | Meta      | 405 B  | Meta's flagship 405 billion parameter language model |
| meta/meta-llama-3-8b-instruct       | Meta      | 8 B    | Meta's 8 billion parameter language model            |
| meta/meta-llama-3-70b-instruct      | Meta      | 70 B   | Meta's 70 billion parameter language model           |


### 6.2 - LLM query function

The **`ask__LLM`** function prepares the query for LLM

Here is abbreviated code

```python
def ask_LLM (question, relevant_docs):
    context = "\n".join(
        [doc['text'] for doc in relevant_docs]
    )

    system_prompt = """
    You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """

    user_prompt = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    ...
    # parameters to the LLM
    params = {
            "top_k": 1,
            "top_p": 0.95,
            "temperature": 0.1,
            "prompt": user_prompt,
            "system_prompt": system_prompt,

            ...
    }
```

### Code explained

**system prompt** guides the models behavior and directing it to generate responses **only** from the provided context.

**User Prompt**: Supplies the **retrieved documents** alongside the user’s question.

**Key Parameters**:
- **top_k=1** - pick the first result from possible results
- **temperature** controls the 'randomness' or 'creativity' of a model.Higher temperatures makes the model more 'creative'.  Smaller values makes the model more deterministic and focused, providing concise and precise answers.

## Step-7: Query LLM


### 7.1 - Query Example

```python
question = "What was the training data used to train Granite models?"
relevant_docs = fetch_relevant_documents(question)
ask_LLM(question=question, relevant_docs=relevant_docs)
```

**Expected Output:**

```text
The Granite Code Instruct models were trained on a combination of permissively licensed data, including the Code Commits Dataset (CommitPackFT) and Math Datasets (MathInstruct and MetaMathQA). Additionally, they were trained on Code Instruction Datasets such as Glaive-Code-Assistant-v3, Self-OSS-Instruct-SC2, Glaive-Function-Calling-v2, and NL2SQL.
```

✅ The model provides a precise response based on the available documents.

### **7.2 - Testing Model Constraints**

Let’s test if our model adheres to instructions when asked about something **outside our dataset**:


```python
question = "When was the moon landing?"
relevant_docs = fetch_relevant_documents(question)
ask_LLM(question=question, relevant_docs=relevant_docs)
```

**Expected Output:**

```text
I'm sorry, the provided context does not contain information about the moon landing.
```

✅ Success! The model correctly limits responses to retrieved documents.

## Conclusion

In this tutorial, we:

🔹 **Performed vector search** to fetch relevant documents.  
🔹 **Queried an LLM** using relevant context.  
🔹 **Validated model constraints** by testing it with an out-of-scope question.

Now, you’re ready to integrate this into your own RAG pipeline!

