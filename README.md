# Bilingual Target-Based Stock Sentiment Dataset (Thai-English)

This repository contains the dataset and source code accompanying the paper:

**_Thai-English Target-Based Stock Sentiment Dataset for Financial News with ICL-Based Evaluation_**  
[To appear in: *Journal/Conference Name*, 2026]  
(*Preprint and DOI will be released soon.*)

**Note**: The dataset is also available on HuggingFace Datasets at:  
[[https://huggingface.co/datasets/DaNDeLioZ/Bilingual_StockTBSA](https://huggingface.co/datasets/airesearch/Bilingual_StockTBSA)]

## Abstract / Motivation

This work introduces a new **bilingual Target-Based Sentiment Analysis (TBSA)** dataset focused on the stock market domain. We collected stock-related financial news from both Thai and international sources, totaling approximately 10,295 Thai and 10,104 English articles.

Each sentence is annotated at the target (TICKER) level using one of six sentiment labels: `positive`, `negative`, `neutral`, `exclude`, `ambiguous`, and `not stock`.

- The `ambiguous` class denotes articles containing both positive and negative impacts on the target stock, making the overall sentiment direction unclear. We excluded this class from the experiments because it contains very few instances in the dataset.

- The `not stock` class corresponds to ticker-like entities (e.g., indices or organizations) erroneously matched during the ticker extraction process. It is included in the dataset statistics for transparency but omitted from model training and evaluation.

**Note**: For detailed definitions and annotation criteria for each sentiment class, please refer to our paper.

We evaluate the dataset using two categories of models:

- Encoder-based models: XLM-RoBERTa-Longformer and mmBERT.

- Large language models: Qwen2.5-72B-Instruct, Llama-3.1-70B-Instruct, DeepSeek-R1-Distill-Llama-70B, Gemma-4-31B-it, and GPT-4o.

We hope this work will be useful for the development of future financial sentiment analysis datasets, as well as for designing effective prompts in financial NLP tasks.

## Repository Structure

```text
.
├── Dataset/
│   ├── Thai_Financial_TBSA_dataset.json            # JSON: Thai financial dataset 2018 - 2023       
│   ├── English_Financial_TBSA_dataset.json         # JSON: English financial dataset 2018 - 2023   
│   └── Tests_ForAnnotatorRecuitment/  
│       └── Recruitment_Test.xlsx                   # A test used for annotator recruitment 
├── Code/
│   ├── Model_finetuning/              
│   │   └── Encoder_finetuning.py       # Finetuning code for encoder models
│   ├── Model_inference_Encoder
│   │   └── Encoder_inference.py        # Inference code for encoder models
│   ├── Model_inference_LLM
│   │   ├── Qwen_Model                  # Inference codes for Qwen2.5-72B-Instruct model (These codes are also used for Llama and Gemma)
│   │   |   ├──Qwen_Zeroshot_Short_inference.py          # Zero-shot short prompt 
│   │   |   ├──Qwen_Zeroshot_Long_inference.py           # Zero-shot long prompt 
│   │   |   ├──Qwen_Fewshot_Vector_inference.py          # 3-shot long prompt with Vector retrieval method 
│   │   |   ├──Qwen_Fewshot_BM25_inference.py            # 3-shot long prompt with BM25 retrieval method 
│   │   |   ├──Qwen_Fewshot_Random_inference.py          # 3-shot long prompt with Random selecting method
│   │   |   ├──Qwen_Fewshot_Hybrid_inference.py          # 3-shot long prompt with Hybrid retrieval method 
│   │   |   └──Qwen_Fewshot_Hardcases_inference.py       # 6-shot long prompt with Hard cases     
│   │   ├── Deepseek_Model               # Inferenc codes for DeepSeek-R1-Distill-Llama-70B model
│   │   |   └── ...          
│   │   ├── GPT4o_Model                  # Inferenc codes for GPT4o model
│   │   |   └── ...     
│   ├── Example_PromptTemplates/                 # Examples of ICL prompts in our experiments
│   ├── Prepare_RetrievedDocuments/         
│   │   |── Prepare_VectorDatabase.py                        # Prepare vector database for a vector retrieval method
|   |   |── Prepare_RetrievedDocument_FromVectorRetriever.py    # Prepare documents retrieved by a vector retriever
|   |   |── Prepare_RetrievedDocument_FromBM25Retriever.py      # Prepare documents retrieved by a BM25 retriever
|   |   |── Prepare_RetrievedDocument_FromRandomRetriever.py    # Prepare documents retrieved by Random sampling
|   |   └── Prepare_RetrievedDocument_FromHybridRetriever.py    # Prepare documents retrieved by a hybrid retriever
|   ├── VLLM-Docker-Deployment            # Docker deployment scripts and configurations for vLLM.
│   │   |── deploy-qwen25-vllm.sh         # Deploy Qwen2.5-72B-Instruct with vLLM using Docker
│   │   |── deploy-llama31-vllm.sh        # Deploy Llama-3.1-70B-Instruct with vLLM using Docker
│   │   |── deploy-deepseekr1-vllm.sh     # Deploy DeepSeek-R1-Distill-Llama-70B with vLLM using Docker
│   │   └── deploy-gemma4-vllm.sh         # Deploy Gemma-4-31B-it with vLLM using Docker
├── requirements.txt
└── README.md
```

**Note**:

- `Example_PromptTemplates/`
  Contains illustrative prompt templates for different in-context learning (ICL) settings.
  The actual prompts, including the retrieved ICL examples, are generated during inference by the code in `Model_inference_LLM/`.  

## Dataset Format

The released dataset is structured in wide format, with one row per news article.

Each article contains:

The full text of the article

A list of target stock mentions and their corresponding sentiment labels

```json
{
  "Article_id": "2",
  "Data-source": "Prachachat-Finance",
  "Date": "2018-01-03",
  "Year": 2018
  "Text": "PACE ออกหุ้นเพิ่มทุน PP จำนวน 400 ล้านหุ้น ให้ SCB มูลค่ารวม 204 ลบ. ผู้สื่อข่าวรายงานว่า บมจ.เพซ ดีเวลลอปเมนท์ คอร์ปอเรชั่น (PACE) ...",
  "Ticker_sentiments": [
    {"ticker": "PACE", "sentiment": "positive"},
    {"ticker": "SCB", "sentiment": "positive"}
  ]
}
```

📌 Label schema:
Although the released dataset includes six sentiment classes (positive, negative, neutral, exclude, ambiguous, not_stock), the experiments reported in our published paper use only four main classes: positive, negative, neutral, and exclude.

🗓️ Temporal splitting:
In our published experiments, we adopt a temporal split for model training and evaluation as follows:

Train: 2018–2020

Validation: 2021

Test: 2022–2023

Users are free to perform their own data splitting as needed for different experimental setups.

💡 Model input:
In practical use, the model receives a (Text, Ticker) pair and predicts the sentiment toward that specific target ticker.
In this released dataset, TICKERs have already been pre-extracted from each article to facilitate reproducible experiments.
For real-world applications, users may need to perform their own ticker extraction step prior to sentiment inference.

## Preparing the Dataset for Our Inference Code

Our inference code is designed to operate on the long format where each row represents a single (text, ticker) pair.

Before running inference on the released dataset, you can convert the dataset to long format using the following code:

```
import pandas as pd

# Load wide-format dataset
df = pd.read_json("Thai_Financial_TBSA_dataset_Updated.json")

# Explode to long format
df_long = df.explode("Ticker_sentiments")
df_long["TICKER"] = df_long["Ticker_sentiments"].apply(lambda x: x["ticker"])
df_long["Sentiment_class"] = df_long["Ticker_sentiments"].apply(lambda x: x["sentiment"])
df_long = df_long.drop(columns=["Ticker_sentiments"])
df_long
```

## Citation

Citation information will be available soon.

```
If you use this dataset or code, please consider citing:
@misc{uthayopas2025tbsa,
  author={Uthayopas, Chayapat
          and Mai-On, Chalermpun
          and Phatthiyaphaibun, Wannaphong
          and Buaphet, Weerayut},
          and Sawatphol, Jitkapat},
          and Sae lim, Sitiporn},
          and Vongkulbhisal, Jayakorn},
          and Vorawathanabuncha, Jasarin},
          and Nutanong, Sarana},
          and Udomcharoenchaikit, Can},
  title = {Thai-English Target-Based Stock Sentiment Dataset for Financial News with ICL-Based Evaluation},
  year = {2025},
  howpublished = {\url{https://github.com/vistec-AI/Bilingual_StockTBSA}},
  note = {Preprint available soon}
}
```

## Acknowledgement
We are grateful for computational resources supported by NSTDA Supercomputer center (ThaiSC) and the National e-Science Infrastructure Consortium for their support of computing facilities for this work. We would like to thank Ms. Lalita Lowphansirikul for the preprocessed financial data and related codes for financial data collection. We also thank our data annotation partner, Wang: Data Market.
