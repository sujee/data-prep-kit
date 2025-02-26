# Preparing PDFs for RAG Pipeline with Data Prep Kit

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing them with relevant, domain-specific information that they wouldn't otherwise know. This tutorial will walk you through preparing PDF documents for a RAG pipeline and running queries on them effectively.

## What You Will Learn

By the end of this tutorial, you’ll understand:

- The fundamentals of RAG and its advantages.
- How to extract and process text from PDFs, including removing duplicate content, chunking documents, and creating embeddings.
- How to store processed data in a vector database.
- How to perform vector searches on documents.
- How to query your documents using an LLM.

## What You Will Need

To follow along, you’ll need:

- A local Python development environment.
- A (free) account at [Replicate](https://replicate.com/) to query LLMs.

## Skill Level

Intermediate

## Estimated Time 

Approximately one hour

## Audience

Python devlopers, LLM application developers, Data Scientists, ML Engineers

---

## Understanding RAG

LLMs are trained on vast amounts of public data but lack access to private or proprietary information. RAG solves this problem by retrieving relevant contextual information and feeding it to the model, improving accuracy and reducing hallucinations.

RAG is widely used for:
- Question-answering systems
- Customer support automation
- Knowledge-based AI applications

For an in-depth understanding, check out [What is Retrieval-Augmented Generation?](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

---


## Workflow Overview

The following diagram illustrates the RAG pipeline workflow:

![](media/rag-overview-2.png)

The tutorial is structured into five steps:

1. **[Getting Started](#step-1-getting-started)** – Set up your environment.
2. **[Processing PDFs](#step-2-processing-pdfs-using-data-prep-kit)** – Extract, clean, and chunk documents.
3. **[Saving Data in a Vector Database](#step-3-storing-data-in-a-vector-database)** – Store processed data efficiently.
4. **[Performing Vector Searches](#step-4-performing-vector-search)** – Retrieve relevant document chunks.
5. **[Querying Documents Using LLMs](#step-5-querying-documents-using-llms)** – Interact with your documents via an LLM.

Let’s dive into each step.


## Step 1: Getting Started
Set up your environment and obtain the necessary code by following this guide:

➡️ [Guide](1-getting-started.md)

---

## Step 2: Processing PDFs Using Data Prep Kit
Here, you’ll:

- Extract text from PDF files.
- Remove duplicate content.
- Split documents into manageable chunks.
- Generate vector embeddings for the chunks.

➡️ [Guide](2-processing-pdfs.md) | [Code](../../examples/notebooks/rag-pdf-1/rag_2_load_data_into_milvus.ipynb)

---

## Step 3: Storing Data in a Vector Database

Once the data is processed, the next step is to store it efficiently in a vector database for fast retrieval.

➡️ [Guide](3-load-data-into-vector-db.md) | [Code](../../examples/notebooks/rag-pdf-1/rag_2_load_data_into_milvus.ipynb)

---

## Step 4: Performing Vector Search

Now, let’s retrieve relevant document chunks by running a vector search.

➡️ [Guide](4-vector-search.md) | [Code](../../examples/notebooks/rag-pdf-1/rag_3_vector_search.ipynb)

---

## Step 5: Querying Documents Using LLMs

The final step is to query your documents using an LLM, allowing you to obtain answers based on the processed PDF data.

➡️ [Guide](5-query-LLM.md) | [Code](../../examples/notebooks/rag-pdf-1/rag_4_query_replicate.ipynb)

## Summary

By completing this tutorial, you’ve learned how to:
- Process and prepare PDFs for a RAG pipeline.
- Store data in a vector database.
- Perform vector searches.
- Query documents using an LLM.

This workflow is a foundation for building AI-powered applications that integrate private knowledge into LLMs efficiently. 

**Happy coding!**

