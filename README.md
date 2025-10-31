# Bilingual Target-Based Stock Sentiment Dataset (Thai-English)

This repository contains the dataset and source code accompanying the paper:

**_Thai-English Target-Based Stock Sentiment Dataset for Financial News with ICL-Based Evaluation_**  
[To appear in: *Journal/Conference Name*, 2025]  
(*Preprint and DOI will be released soon.*)

**Note**: The dataset is also available on HuggingFace Datasets at:  
[[https://huggingface.co/datasets/DaNDeLioZ/Bilingual_StockTBSA](https://huggingface.co/datasets/airesearch/Bilingual_StockTBSA)]

## Abstract / Motivation

This work introduces a new **bilingual Target-Based Sentiment Analysis (TBSA)** dataset focused on the stock market domain. We collected stock-related financial news from both Thai and international sources, totaling approximately 10,300 Thai and 10,120 English articles.

Each sentence is annotated at the target (TICKER) level using one of six sentiment labels: `positive`, `negative`, `neutral`, `exclude`, `ambiguous`, and `not stock`.

- The `ambiguous` class denotes unclear sentiment polarity but was excluded from modeling due to its rarity.

- The `not stock` class corresponds to ticker-like entities (e.g., indices or organizations) erroneously matched during the ticker extraction process. It is included in the dataset statistics for transparency but omitted from model training and evaluation.

We evaluate the dataset using both encoder-based model (e.g., XLM-RoBERTa-Longformer) and large language models (i.e., Qwen2.5-72B-Instruct, GPT4o, LLaMA-3.1-70B-Instruct).

We hope this work will be useful for the development of future financial sentiment analysis datasets, as well as for designing effective prompts in financial NLP tasks.

## Repository Structure

```text
.
├── Dataset/
│   ├── Thai_Financial_TBSA_dataset.json            # JSON: Thai financial dataset 2018 - 2023       
│   ├── English_Financial_TBSA_dataset.json         # JSON: English financial dataset 2018 - 2023   
│   └── Tests_ForAnnotatorRecuitment/  
│       └── Recruitment_Test.xlsx                   # A test used for annotator recruitment 
│
├── Code/
│   ├── Model_finetuning/              
│   │   └── Encoder_finetuning.py       # Finetuning code for encoder models
│   ├── Model_inference_Encoder
│   │   ├── Encoder_inference.py        # Inference code for encoder models
│   ├── Model_inference_LLM
│   │   ├── Qwen_Zeroshot_Short_inference.py          # Zero-shot short prompt (Qwen2.5-72B-Instruct)
│   │   ├── Qwen_Zeroshot_Long_inference.py           # Zero-shot long prompt (Qwen2.5-72B-Instruct)
│   │   ├── Qwen_Fewshot_Vector_inference.py          # 3-shot long prompt with Vector retrieval method 
│   │   ├── Qwen_Fewshot_BM25_inference.py            # 3-shot long prompt with BM25 retrieval method 
│   │   ├── Qwen_Fewshot_Random_inference.py          # 3-shot long prompt with Random selecting method 
│   │   ├── Qwen_Fewshot_Hardcases_inference.py       # 6-shot long prompt with Hard cases 
│   │   ├── GPT4o_Zeroshot_Short_inference.py         # Inference code for GPT4o model
│   │   ├── GPT4o_Zeroshot_Long_inference.py          
│   │   ├── GPT4o_Fewshot_Vector_inference.py        
│   │   ├── GPT4o_Fewshot_BM25_inference.py          
│   │   ├── GPT4o_Fewshot_Random_inference.py       
│   │   └── GPT4o_Fewshot_Hardcases_inference.py     
│   ├── Examples_PromptTemplate/                      # Example of ICL prompt template
│   ├── Prepare_VectorDatabase/         
│   │   └── Prepare_VectorDatabase.py                 # Prepare vector database for few-shot retrieval
│   └── requirements.txt
└── README.md
```

**Note**:

- `Model_inference_LLM/`  
  Python codes for in-context learning (ICL) experiments with large language models (e.g., Qwen2.5-72B-Instruct, GPT4o, LLaMA-3.1-70B-Instruct). LLaMA model uses the same code as Qwen model, with only the model and tokenizer changed.

- `Example_PromptTemplate/`  
  Includes example prompt templates used in different ICL scenarios, shown for clarity.  
  The actual prompts used during inference are implemented in the code provided in `Model_inference_LLM/`.

## Dataset Format

The released dataset is structured in wide format, with one row per news article.

Each article contains:

The full text of the article

A list of target stock mentions and their corresponding sentiment labels

```json
{
  "Article_id": "1",
  "Data-source": "Prachachat",
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

## Using with Our Inference Code

Our provided inference code (see inference_qwen.py) is designed to operate on the long format where each row corresponds to a single (text, ticker) pair.

If you wish to use our code with the released dataset, you can convert the dataset to long format using the following snippet:

```
import pandas as pd

# Load wide-format dataset
df = pd.read_json("Thai_Financial_TBSA_dataset.json")

# Explode to long format
df_long = df.explode("Target_sentiment")
df_long["TICKER"] = df_long["Target_sentiment"].apply(lambda x: x["ticker"])
df_long["Sentiment_class"] = df_long["Target_sentiment"].apply(lambda x: x["sentiment"])
df_long = df_long.drop(columns=["targets"])
```


## Citation

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
  howpublished = {\url{https://github.com/anonymous/tbsa-th}},
  note = {Preprint available soon}
}
```

## Acknowledgement
This research was supported in part by the Thailand Capital Market Development Fund (CMDF), the WangchanX Project, Siam Commercial Bank (SCB), SCB X Public Company Limited, and PTT Public Company Limited.
We would like to thank Ms. Lalita Lowphansirikul for the preprocessed financial data and related codes for financial data collection. We also thank our data annotation partner, Wang: Data Market.
