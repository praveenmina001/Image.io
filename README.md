# Multi-Agent RAG for Multimodal Documents

## Project Description

This project implements a modular, multi-agent Retrieval-Augmented Generation (RAG) pipeline that improves factual accuracy and reduces hallucinations in question-answering over documents containing both text and images. The system detects and classifies images, avoids forced text conversion, and incorporates fallback mechanisms for unprocessable content.

## Features

- Modular agents for document segmentation, image detection, retrieval, verification, and response generation.
- Intelligent image handling: Only processes images if relevant, flags uncertainty when unprocessable.
- Hallucination mitigation: Abstains or defers when context is insufficient, never fabricates.
- Evaluation-ready: Includes test cases and evaluation script.

## System Architecture

1. **Document Segmenter:** Splits documents into sections, keeps track of images.
2. **Image Relevance Agent:** Checks if images are important for the question.
3. **Image Handler:** Tries to process images if needed, otherwise flags for review.
4. **Retriever:** Finds the best matching section for the query.
5. **Verifier:** Checks if enough information is present.
6. **Response Generator:** Answers the question or honestly admits uncertainty.


