# 2 - Processing PDFs

In this stage, we will process PDF documents to prepare them for Retrieval Augmented Generation (RAG) queries. The steps involved are as follows:

1. Reading and extracting contents from PDFs
2. Removing duplicates
3. Splitting the extracted content into chunks
4. Creating embeddings for the chunks

You can find the code for this process in [rag_1_dpk_process_python.ipynb](../../examples/notebooks/rag-pdf-1/rag_1_dpk_process_python.ipynb)


## Step-1: Configuration

We have a common configuration parameters defined in [my_config.py](../../examples/notebooks/rag-pdf-1/my_config.py).  Below is a snippet:


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
MY_CONFIG.MAX_CONTEXT_WINDOW = 4096 #tokens
```

This configuration class centralizes all parameters, making it easy to add new ones:


```python
MY_CONFIG.SAMPLE_CONFIG = "hello"
```

In the main code, we initialize it as:

```python
from my_config import MY_CONFIG
```

### Best Practice Tip 💡

Having all our configuration parameters confined to one class keeps the global namespace clean.


## Step-2: Downloading Data

### Step 2.1 - Fetching PDFs

We download three PDF documents from [arxiv.org](https://arxiv.org), using the `download_file` function from [utils.py](../../examples/notebooks/rag-pdf-1/utils.py). The function checks for local existence before downloading.

```python
from utils import download_file

download_file (url = 'https://arxiv.org/pdf/1706.03762', local_file = os.path.join(MY_CONFIG.INPUT_DATA_DIR, 'attention.pdf' ))
download_file (url = 'https://arxiv.org/pdf/2405.04324', local_file = os.path.join(MY_CONFIG.INPUT_DATA_DIR, 'granite.pdf' ))
download_file (url = 'https://arxiv.org/pdf/2405.04324', local_file = os.path.join(MY_CONFIG.INPUT_DATA_DIR, 'granite2.pdf' )) # duplicate
```

**Why the duplicate?**  
We intentionally download the same file twice to demonstrate duplicate removal in a later step.


## Step 2.2 - Setting Up Input and Output Directories

Processing follows a multi-stage approach:


`input ---(stage-1)--> output/01_stage1_out ---(stage 2)---> output/02_stage2_out` 

The output (`output/01_stage1_out`) becomes input to the next stage.  And so on.

The final processing pipeline:

`pdfs → (extract text) → 01_parquet_out → (dedupe) → 02_dedupe_out 
     → (chunking) → 03_chunk_out → (embeddings) → 04_embeddings_out → output_final`

Output folder structure:

```
output
├── 01_parquet_out
├── 02_dedupe_out
├── 03_chunk_out
├── 04_embeddings_out
└── output_final

```

### 📌 Why Parquet format?

The Data Prep Kit uses **Parquet** for intermediate data storage due to its efficiency:

✅ **Compressed storage** reduces disk usage.  
✅ **Faster queries** enable efficient data access.  
✅ **Scales well** for large datasets.

Learn more:  

* [Apache Parquet Official Site](https://parquet.apache.org/)
* [Databricks: What is Parquet?](https://www.databricks.com/glossary/what-is-parquet)

## Step-3: Extracting Contents from PDF (PDF2Parquet)

### 3.1 : Extracting text

We use the `Pdf2Parquet` transformation to extract text from PDFs.

- Class: `dpk_pdf2parquet.transform_python.Pdf2Parquet`
- Found in  package : `data-prep-toolkit-transforms[pdf2parquet]` (installed as part of `data-prep-toolkit-transforms[all]`)

[PDF2Parquet documentation](https://github.com/IBM/data-prep-kit/tree/dev/transforms/language/pdf2parquet)

Here is the code that does it (abbreviated):

```python
from dpk_pdf2parquet.transform_python import Pdf2Parquet
from dpk_pdf2parquet.transform import pdf2parquet_contents_types

result = Pdf2Parquet(input_folder= 'input',
                    output_folder= 'output/01_parquet_out',
                    data_files_to_use=['.pdf'],
                    pdf2parquet_contents_type=pdf2parquet_contents_types.MARKDOWN,   # markdown
                    ).transform()

if result != 0:
    raise Exception ('process failed')
```

Parameters explained:
- **`input_folder`** : where to read data files from
- **`output_folder`** : destination for output folder
- **`data_files_to_use`** : specify file filters; here we are only intereseted in PDF files
- **`pdf2parquet_contents_type`** : This controls the output format for extracted content.  It can be JSON or markdown.
    - for markdown: `pdf2parquet_contents_type=pdf2parquet_contents_types.MARKDOWN`
    - for JSON: `pdf2parquet_contents_type=pdf2parquet_contents_types.JSON'`

### 3.2 - Understanding the Output

Each input PDF generates a corresponding Parquet file in `output/01_parquet_out`.

```
├── 01_parquet_out
│   ├── attention.parquet
│   ├── granite2.parquet
│   ├── granite.parquet
│   └── metadata.json
```


Let's inspect the output using `read_parquet_files_as_df`  - a utility function from [utils.py](../../examples/notebooks/rag-pdf-1/utils.py)

```python
from utils import read_parquet_files_as_df

output_df = read_parquet_files_as_df('01_parquet_out')
output_df.head(5)
```
<img src="media/pdf2pq-output1.png" style="max-width:90%;"/>

Output explained:

- we will have one entry per input file (so 3 total)
- **`filename`** column is the name of input file. 
- **`contents`** column has the markdown text extracted from PDFs
- **`num_pages`**, **`num_tables`** columns indicate detected metadata
- **`document_id`** is a unique id generated for each document
- **`document_hash`** is a hash calcuated on entire file
- **`hash`** is calculated from **contents** column.  Note this is different from `document_hash`
- **`pdf_convert_time`** denotes the time took to process each pdf in seconds

## Step-4: Eliminating Duplicate Documents

We have 2 identical copies of `granite.pdf`.  Also observe the `hash` values for these documents are the same.  We are going to eliminate one of the duplicates.

<img src="media/duplicates-1.png" style="max-width:90%;"/>

Removing duplicates has benefits such as

- reducing the amount of data we have to process down the pipeline
- improving the quality of RAG results

### 4.1 - Removing Duplicates

We will use **Exect Deduplicate transform (Ededupe)** for this.  Ededupe compute the hash values on `contents` of each file.  Files with duplicate hashes are filtered out.

- Class : `dpk_ededup.transform_python.Ededup`
- Found in  package : `data-prep-toolkit-transforms[ededup]` (installed as part of `data-prep-toolkit-transforms[all]`)

[Exact dedeupe filter documentation](https://github.com/IBM/data-prep-kit/blob/dev/transforms/universal/ededup/README.md)

```python
from dpk_ededup.transform_python import Ededup

result = Ededup(input_folder='01_parquet_out',
                output_folder='output/02_dedupe_out',
                ededup_doc_column="contents",
                ededup_doc_id_column="document_id"
                ).transform()

if result != 0:
    raise Exception ('process failed')
```

Parameters explained:

- **`input_folder`**: we are reading the output created by previous step (pdf2parquet)
- **`output_folder`**: filtered files will be in 'output/02_dedupe_out'
- **`ededup_doc_column = 'contents'`**: This is the column we will calculate hash values for
- **`ededup_doc_id_column = 'document_id'`**: column to store the computed hash above

### 4.2 - Understanding the Output

```python
from utils import read_parquet_files_as_df

output_df = read_parquet_files_as_df('output/02_dedupe_out')
output_df.head(5)
```

After deduplication, one of the duplicate `granite.pdf` files is removed.


<img src="media/ededupe-output1.png" style="max-width:90%;"/>



##  Step-5: Splitting Documents into Chunks

### 5.1 - Chunking Text

We split documents into smaller **retrieval-friendly** chunks using **`DocChunk`** transformation.

- Class : `dpk_doc_chunk.transform_python.DocChunk`
- Found in  package : `data-prep-toolkit-transforms[doc_chunk]` (installed as part of `data-prep-toolkit-transforms[all]`)

[Chunking transform documentation](https://github.com/IBM/data-prep-kit/blob/dev/transforms/language/doc_chunk/README.md)

```python
from dpk_doc_chunk.transform_python import DocChunk

result = DocChunk(input_folder='output/02_dedupe_out',
                    output_folder='output/03_chunk_out',
                    doc_chunk_chunking_type= "li_markdown",
                    # doc_chunk_chunking_type= "dl_json",
                    doc_chunk_chunk_size_tokens = 128,  # default 128
                    doc_chunk_chunk_overlap_tokens=30   # default 30
                    ).transform()
```

Parameters explained:

- **`input_folder`** and `output_folder` are self explanatory
- **`doc_chunk_chunking_type = "li_markdown"`** as we are parsing markdown text.  Here we will be using Llama-Index markdown parser.  If we want to split json text into chunks set this to `"dl_json"`
- **`doc_chunk_chunk_size_tokens = 128`** : Size of each chunk is about 128 tokens (approx 100 words).  Default value for this is 128
- **`doc_chunk_chunk_overlap_tokens=30`** : overlap between chunks is about 30 tokens (~25 words).   Default value for this is 30.

### 📌 Chunking Strategy

We are using  **chunking size = 128 tokens (~100 words)** and **Overlap: 30 tokens (~25 words)**  

**Experiment with chunking settings to find one that works best for your documents**

🔗 **Further reading on chunking:**  

- [A Guide to Chunking Strategies for Retrieval Augmented Generation (RAG)](https://zilliz.com/learn/guide-to-chunking-strategies-for-rag)
- [Understanding Chunking Strategies](https://www.sagacify.com/news/a-guide-to-chunking-strategies-for-retrieval-augmented-generation-rag)

### 5.2 - Understanding chunking output

```python
from utils import read_parquet_files_as_df

input_df = read_parquet_files_as_df('output/02_dedupe_out')
output_df = read_parquet_files_as_df('output/03_chunk_out')

print (f"Files processed : {input_df.shape[0]:,}")
print (f"Chunks created : {output_df.shape[0]:,}")

output_df.sample(min(3, output_df.shape[0]))
```

Output 

```text
Files processed : 2
Chunks created : 60
```

The 2 PDF files have been split into 60 chunks.

Here is a sample output of few chunks

<img src="media/chunks-output-1.png" style="max-width:90%;"/>


## Step-6: Create Embeddings for Chunks

The final step in our processing pipeline is to **generate embeddings for the extracted chunks**.  

### 📌 Why embeddings?


**Embeddings** play a crucial role in the RAG process, enabling fast search and retrieval of complex data like text and images.  

📖 **Further Reading**: [Embeddings Explained](embeddings-explained.md)



### 6.1 - Calculate Embeddings

We generate embeddings for each chunk using `TextEncoder`.

- Class : `dpk_text_encoder.transform_python.TextEncoder`
- Found in  package : `data-prep-toolkit-transforms[text_encoder]` (installed as part of `data-prep-toolkit-transforms[all]`)

[Embeddings / Text Encoder documentation](https://github.com/IBM/data-prep-kit/blob/dev/transforms/language/text_encoder/README.md)



Here is the code (abbreviated) to transform **chunks --> embeddings**


```python
from dpk_text_encoder.transform_python import TextEncoder

result = TextEncoder(input_folder= 'output/03_chunk_out', 
                    output_folder= 'output/04_embeddings_out', 
                    text_encoder_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                    ).transform()
```

Parameters:

- **`input_folder` and `output_folder`** are respectively for reading and writing 
- **`text_encoder_model_name = 'sentence-transformers/all-MiniLM-L6-v2'`** - Here we can specify any open source embedding model.  There are numerous embedding models we can use.  We are using **sentence-transformers/all-MiniLM-L6-v2** as it is a small model (hence quick to run) and gives decent results. [Hugging Face's embedding model leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is an excellent resource for finding suitable embedding models.

### 6.2 - Understanding embedding output

```python
from utils import read_parquet_files_as_df

output_df = read_parquet_files_as_df('output/04_embeddings_out')
output_df.sample(3)
```

In the produced output we see a new column called **embeddings**.  This column has embeddings computed for text **contents** column - these chunked texts

<img src="media/embeddings-3-output.png" style="max-width:90%;"/>

## Conclusion

**Summary of the Processing Pipeline**  

✅ **Extract text** from PDFs → Store in Parquet.  
✅ **Deduplicate** documents → Remove redundant files.  
✅ **Chunk documents** → Optimize for retrieval.  
✅ **Generate embeddings** → Prepare for vector search.

Now that our PDFs have been processed  the next step is to **import the embeddings into a vector database** for efficient querying.

👉 Continue to the next step : [import data into vector database](3-load-data-into-vector-db.md)