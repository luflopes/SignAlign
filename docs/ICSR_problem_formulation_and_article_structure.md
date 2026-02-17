# Identity-Conditioned Signature Retrieval (ICSR)

## Problem Formulation and Article Structure Proposal

------------------------------------------------------------------------

# 1. Problem Formulation

## 1.1 Motivation

In real-world document processing pipelines, documents frequently
contain **multiple signatures** from different individuals (e.g.,
client, witness, manager, lawyer).

Traditional approaches focus on: - Signature detection - Signature
cleaning - 1:1 signature verification

However, they do **not address the following operational bottleneck**:

> Given a document with multiple detected signatures, how do we
> automatically identify which signature belongs to a specific person?

This gap motivates the formalization of a new task.

------------------------------------------------------------------------

## 1.2 Proposed Task

### Identity-Conditioned Signature Retrieval (ICSR)

Given:
- A document $D$
- A set of detected signature candidates $S = \{s_1, ..., s_n\}$
- A textual identity $t$ (person name)

The goal is to retrieve the correct signature:

$$
\hat{s} = \arg\max_{s_i \in S} \mathrm{Sim}(f(s_i), g(t))
$$

Where:
- $f(\cdot)$ is the visual encoder  
- $g(\cdot)$ is the textual encoder  
- $\mathrm{Sim}$ is cosine similarity in embedding space


------------------------------------------------------------------------

# 2. Core Contributions

1.  **New Problem Definition**\
    Formalization of Identity-Conditioned Signature Retrieval in
    multi-signature documents.

2.  **First Public Dataset for Text--Signature Alignment**\
    \~4,500 (image, name) pairs\
    LLM-assisted annotation with human validation\
    Individual-based split to prevent identity leakage

3.  **Realistic Multi-Signature Evaluation Protocol**\
    Retrieval within document-level candidate sets\
    Metrics: Accuracy@k, MRR, Mean Rank

4.  **Cross-Dataset and Cross-Language Generalization**\
    Training on English names\
    Evaluation on unseen Portuguese (PT-BR) dataset

5.  **Explainability Analysis**\
    Gradient-based visualization demonstrating semantic alignment
    between textual identity and signature regions.

------------------------------------------------------------------------

# 3. Proposed Paper Structure

## 1. Introduction

-   Real-world multi-signature scenario
-   Limitations of existing signature verification approaches
-   Formalization of ICSR
-   Summary of contributions

------------------------------------------------------------------------

## 2. Related Work

### 2.1 Signature Verification

Classical 1:1 matching approaches

### 2.2 Writer Identification

Closed-set identification, not conditioned on textual identity

### 2.3 Vision-Language Models

CLIP, SigLIP, multimodal embedding learning

### 2.4 Document AI Pipelines

Gap in intra-document signature disambiguation

------------------------------------------------------------------------

## 3. Problem Definition

-   Formal mathematical definition
-   Difference from verification
-   Retrieval-based evaluation metrics

------------------------------------------------------------------------

## 4. SigAlign Dataset

-   Data sources
-   Annotation pipeline (LLM + manual review)
-   Statistics:
    -   Number of documents
    -   Signatures per document distribution
    -   Rejection rate
-   Identity-based split
-   Example samples

------------------------------------------------------------------------

## 5. Methodology

-   Vision-Language architectures (CLIP, TinyCLIP, SigLIP)
-   Preprocessing pipeline
-   Loss functions (InfoNCE, Triplet, Combined)
-   Data augmentation strategy

------------------------------------------------------------------------

## 6. Experiments

### 6.1 Experimental Setup

Hyperparameters, training protocol

### 6.2 Fine-Tuning vs Frozen

### 6.3 Loss Function Analysis

### 6.4 Embedding Gap Analysis

### 6.5 Multi-Signature Retrieval Evaluation (Main Section)

### 6.6 Cross-Dataset and Cross-Language Evaluation

------------------------------------------------------------------------

## 7. Explainability Analysis

-   Gradient-based attention visualization
-   Correct vs incorrect retrieval comparison

------------------------------------------------------------------------

## 8. Discussion

-   Implications for Document AI pipelines
-   Strengths and limitations
-   Generalization insights
-   Future work directions

------------------------------------------------------------------------

## 9. Conclusion

-   Introduction of ICSR task
-   Public dataset release
-   Empirical validation
-   Demonstration of practical applicability

------------------------------------------------------------------------

# 4. Strategic Positioning

To strengthen acceptance probability:

-   Emphasize novelty of the task definition
-   Highlight realistic evaluation protocol
-   Stress cross-language robustness
-   Clearly distinguish retrieval from verification
-   Provide detailed dataset statistics and candidate distribution

------------------------------------------------------------------------

# Target Venues

-   IJDAR (International Journal on Document Analysis and Recognition)
-   Pattern Recognition Letters
-   ICDAR (next cycle submission)
-   Relevant Document AI workshops (CVPR/ECCV/ICCV)

------------------------------------------------------------------------

Prepared for advisor discussion.
