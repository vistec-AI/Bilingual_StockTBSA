"""
LLM-based inference script for Stock TBSA (Target-Based Sentiment Analysis)

This script performs Fewshot inference using a large language model (LLM)
(e.g., Qwen2.5-72B-Instruct) on a TBSA test set and outputs predicted sentiment labels.
"""

# -----------------------------
# Standard library imports
# -----------------------------
import os
import time
import json
from typing import List

# -----------------------------
# Third-party imports
# -----------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum, EnumMeta
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI


# -----------------------------
# Prompt template and label enums
# -----------------------------
# 🔹NOTE:
# - This is the 6-shot prompt with hard cases in English (Translated from Thai news) used for inference.
# - A Thai version is available at: `Code/Examples_PromptTemplate/6-shot_LongPrompt_Hardcases_Thai.txt`

PROMPT_TEMPLATE = """I want you to act as a financial expert and NLP researcher in the field of data-centric research.
 
I want you to annotate stock sentiment for each stock TICKER that is mentioned in an input document.
Read the entire input document and assign final stock sentiment to each target stock TICKER.
-Sentiment must be determined solely based on the content of the given document.
-Respond only in Thai.
 
These are definitions for stock sentiment classes.
- Positive class = "The content of this news article has a positive impact on the target stock."
- Negative class = "The content of this news article has a negative impact on the target stock."
- Neutral class = "The content of this news article has neither a positive nor negative impact on the target stock."
- Exclude class = "The content of this news article does not fall into the above three classes or is unrelated to the target stock in terms of investment."
 
Annotation rules:
- If the news article includes stock direction analysis by analysts, assign the sentiment based on the analyst's opinion.
- If the company associated with the target stock engages in business expansion, increases production capacity, acquires other businesses, or signs cooperation agreements with other companies, label that stock as Positive.
- If the company associated with the target stock adjusts its policies to enhance business opportunities, label that stock as Positive.
- If the company associated with the target stock is involved in a lawsuit but wins the case without incurring damages, label that stock as Positive.
- If investors make additional investments or purchase the target stock for their portfolio, indicating that the stock is performing well, label that stock as Positive.
- If the target stock is included in the SET50 or SET100 index, indicating business growth, label that stock as Positive.
- If the target stock is involved in a lawsuit and loses the case, requiring payment or damages, label that stock as Negative.
- If the target stock is subject to financial audits or required to submit financial statements for review, label that stock as Negative.
- If investors sell the target stock from their portfolios, indicating potential issues, label that stock as Negative.
- If the target stock is placed under a cash balance restriction or has an extension of such a restriction, label that stock as Negative.
- If the stock exchange issues a trading alert for the target stock, label that stock as Negative.
- If a court or a stock exchange orders the company associated with the stock to provide clarification, indicating no wrongdoing, label that stock as Neutral.
- If a court announces an extension of the case deliberation period, indicating no wrongdoing, label that stock as Neutral.
- If the company associated with the stock receives a warning but no penalty, label that stock as Neutral.
- If the company associated with the stock files a lawsuit against another company, label the suing company as Neutral.
- Reports on opening or closing stock prices for a single day should be labeled as Neutral because single-day price changes do not indicate long-term stock trends.
- If the article discusses a stock split or reverse stock split, label that stock as Neutral.
- If the article mentions "buying pressure" or "selling pressure," label that stock as Neutral since these are short-term fluctuations, and the market may normalize the next day.
- If the article discusses "bond issuance," label the company associated with the stock as Neutral, unless the article states that the bond issuance is funding a new project, in which case label that stock as Positive.
- For banks or companies underwriting bonds for others, label the underwriters as Neutral.
- Articles about loan approvals should label the company receiving the loan as Positive, while labeling the approving bank as Neutral.
- Articles discussing adjustments to gasoline or gasohol prices should be labeled as Neutral.
- If a company or its stock is mentioned briefly in an article without positive or negative elaboration, label that stock as Neutral.
- Articles about donations, promotions, campaigns, annual or extraordinary general meetings, receiving or giving awards, recreational activities, non-investment-related activities, changes in executive positions (appointments, resignations, retirements), IPO launches, meeting venue changes, product launches as advertisements, or factory/company visits should be labeled as Exclude.
- Participating in exhibitions, showcasing innovations, launching new products, or releasing new packages is considered public relations and should be labeled as Exclude.
- Articles discussing overall market economic conditions without specific stock references should be labeled as Exclude.
- Articles unrelated to stock investments should be labeled as Exclude.
- Articles about application system maintenance should be labeled as Exclude.
- If the target stock is mentioned in the context of its first trading day, as this does not reflect long-term performance, label that stock as Exclude.
- Articles mentioning Biglot transactions should be labeled as Exclude.
- Articles discussing dividend payouts should be labeled as Exclude.
- Articles about the biography of company executives, unrelated to stock investments, should be labeled as Exclude.
- If the company associated with the target stock is acting as a stock analyst or reporting news about other stocks, label that stock as Exclude.

In cases of company transactions where a business entity is bought or sold and both the buyer and seller benefit (e.g., divestment of subsidiaries or business units), the stock should be labeled as Positive.
EXAMPLE: BTS sells land-holding business to NOBLE worth 2.3 billion baht. BTS sells subsidiary “Future Domain,” which operates a land-holding business, to NOBLE worth 2.3 billion baht. BTS sells subsidiary “Future Domain,” which operates a land-holding business, to NOBLE worth 2.3 billion baht. On July 2, 2021, BTS Group Holdings Public Company Limited, or BTS, disclosed that Kingkaew Assets Company Limited, a wholly owned subsidiary of the Company, had sold all of its investment in Future Domain Company Limited, which is a wholly owned subsidiary of Kingkaew Assets Company Limited. As a result, Future Domain Company Limited has ceased to be a subsidiary of the Company. The divestment of this investment consisted of the sale of all investments in Future Domain Company Limited, which operates a land-holding business and owns land located in areas close to residential communities and shopping malls, by selling (1) all 1,000,000 shares with a par value of 100 baht per share, representing 100 percent of the total shares in Future Domain Company Limited, and (2) all promissory notes issued by Future Domain Company Limited. According to the share purchase agreement between Kingkaew Assets Company Limited and Noble Development Public Company Limited, or NOBLE, the total transaction value amounted to 2,298,613,000.00 baht. The agreed price was mutually determined by both parties based on the market value of the assets and the value of liabilities according to the financial statements prepared by the management (internal accounts) of Future Domain Company Limited.
TICKER: NOBLE
SENTIMENT_CLASS: positive

Purchasing or accumulating stocks into a portfolio indicates that the company is performing well; therefore, the stock should be labeled as Positive.
EXAMPLE: "Dr. Chalieo" spends over 2 million baht to acquire another 500,000 shares of EPG. On June 4, 2019, reporters stated that, according to Form 59, a report of changes in securities holdings and futures contracts by executives submitted to the Securities and Exchange Commission (SEC), Dr. Chalieo Vitoorapakorn, Deputy Chief Executive Officer of Eastern Polymer Group Public Company Limited (EPG), purchased 500,000 shares on May 31, 2019, at an average price of 5.15 baht per share, totaling approximately 2.58 million baht. It is noted that on May 30, 2019, Dr. Chalieo Vitoorapakorn had already purchased 1,000,000 EPG shares at an average price of 5.05 baht per share, totaling approximately 5.05 million baht.
TICKER: EPG
SENTIMENT_CLASS: positive

Reports on opening or closing stock prices for a single day should be labeled as Neutral because single-day price changes do not indicate long-term stock trends.
EXAMPLE: SET closed the morning session up 6.64 points, with PTT recording the highest trading value of 1.30 billion baht. SET closed the morning session up 6.64 points, with PTT recording the highest trading value of 1.30 billion baht. The index closed this morning at 1,772.57 points, an increase of 6.64 points or 0.38%, with a trading value of 24.1 billion baht. On May 14, 2018, reporters reported that the Stock Exchange of Thailand (SET) index closed this morning (May 14) at 1,772.57 points, up 6.64 points or 0.38%, with a trading value of 24,096.86 million baht. The top five securities with the highest trading value were PTT with 1,297.50 million baht, closing at 56.75 baht, unchanged; WHA with 1,095.53 million baht, closing at 4.16 baht, up 0.10 baht; BEAUTY with 948.23 million baht, closing at 21.50 baht, up 0.10 baht; EA with 879.86 million baht, closing at 36.50 baht, up 1 baht; and BANPU with 788.82 million baht, closing at 20.40 baht, unchanged.
TICKER: PTT
SENTIMENT_CLASS: neutral

Product or application launches serve as public relations activities. Therefore, such news should be labeled as Exclude.
EXAMPLE: BEAUTY launches “Skin-Nourishing Vitamins” to penetrate the beauty retail market. BEAUTY introduces new products under the Beauty Buffet brand, namely “Skin-Nourishing Vitamins,” consisting of three items, entering the beauty retail market with sales starting on July 1, 2021. On June 30, 2021, Beauty Community Public Company Limited, or BEAUTY, a distributor of products under the Beauty Buffet brand, unveiled three new skin-nourishing products: BEAUTY BUFFET MULTIVITAMIN BODY BRIGHT SHOWER SERUM, BEAUTY BUFFET MULTIVITAMIN AFTER BATH BODY ESSENCE, and BEAUTY BUFFET VITAMIN C AURA SOAP. These products are designed to cleanse and nourish the skin, helping to eliminate dullness and damage, lock in brightness, and reveal beautiful skin. Enriched with vitamins, they visibly enhance radiance and naturally moisturize the skin at an affordable price. Sales will begin on July 1, 2021, through consumer product channels and major and minor beauty product retail stores nationwide, totaling more than 16,000 outlets.
TICKER: BEAUTY 
SENTIMENT_CLASS: exclude

News related to the Thai Baht exchange rate or interest rate adjustments is not directly related to stock investment decisions and should be labeled as Exclude.
EXAMPLE: KBANK projects baht range next week at 31.15–31.50; advises close watch on economic data and domestic COVID-19. Kasikornbank Public Company Limited (KBANK) expects the Thai baht to move within the range of 31.15–31.50 baht per U.S. dollar during the upcoming week (May 31–June 4, 2021). Key factors to monitor include the current account balance and the April Monetary Policy Report from the Bank of Thailand, as well as the domestic COVID-19 situation. Important U.S. economic data to be released include non-farm payrolls, the unemployment rate, private-sector employment from ADP, the May PMI/ISM indexes for manufacturing and services, April construction spending, weekly jobless claims, and the Federal Reserve’s Beige Book. In addition, markets are awaiting the G-7 meeting, developments in the global COVID-19 situation, and the May manufacturing and services PMI indexes from China, the Eurozone, and the United Kingdom. Over the past week, the baht strengthened slightly, moving in line with the yuan and most other regional currencies. The currency also received additional support from net purchases of Thai bonds by foreign investors and Thailand’s April export data, which showed stronger-than-expected growth. Meanwhile, the U.S. dollar came under pressure during the week after senior Federal Reserve officials reiterated that the Fed would maintain its accommodative monetary policy stance and emphasized that U.S. inflation would likely be only temporary.
TICKER: KBANK 
SENTIMENT_CLASS: exclude

News about IPO announcements or IPO trading activities should be labeled as Exclude.
EXAMPLE: NRF shows strong performance, reaching 8 baht within 5 minutes of the Pre-Open session, 74% above its IPO price of 4.60 baht. The total trading volume before the market opened was approximately 51.74 million shares, with buy orders (Bid) of about 39.82 million shares and sell orders (Offer) of about 32.92 million shares. Siam Commercial Bank Public Company Limited (SCB) served as the financial advisor. On October 9, 2020, it was reported that the share price of NR Instant Produce (NRF) stood at 8 baht during the Pre-Open session, 5 minutes before the market opened, compared to the IPO price of 4.60 baht. The trading volume before the market opened was approximately 51.74 million shares, with buy orders (Bid) of about 39.82 million shares and sell orders (Offer) of about 32.92 million shares. Siam Commercial Bank Public Company Limited (SCB) acted as the financial advisor. NRF operates in the business of producing, sourcing, and distributing seasoning products, ready-to-eat foods, cooking ingredients, vegan food products without eggs or dairy, plant-based protein foods, and instant powdered and liquid beverages. The company offered 340 million shares in its initial public offering (IPO) at the price of 4.60 baht per share during September 28 – October 5, 2020.
TICKER: NRF 
SENTIMENT_CLASS: exclude

TARGET_ARTICLE: {doc}
TICKER: {tags}
SENTIMENT_CLASS: 
"""

# -----------------------------
# Configuration (replace with your actual paths)
# -----------------------------
TEST_JSON_PATH = "./path/to/test_data.json"  # <-- Replace with your test set path
SAVE_JSON_PATH = "./path/to/output_predictions.json"  # <-- Output predictions
SAVE_TIME_PATH = "./path/to/inference_timing.txt"  # <-- Output timing info

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # can change to other LLM such as "meta-llama/Llama-3.1-70B-Instruct"
API_KEY = "EMPTY"  # <-- Replace with your actual API key
API_BASE = "https://your-inference-endpoint.com/v1"  # <-- Replace with your API base
TEMPERATURE = 0.0


class EnumDirectValueMeta(EnumMeta):
    def __getattribute__(cls, name):
        value = super().__getattribute__(name)
        if isinstance(value, cls):
            value = value.value
        return value


class SentimentType(Enum, metaclass=EnumDirectValueMeta):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    EXCLUDE = "exclude"


class Sentiment(BaseModel):
    sentiment: List[SentimentType]
    # reason: Optional[str] = Field(description="Justification for the label")  # Optional if needed


# -----------------------------
# Helper functions
# -----------------------------
def create_prompt(doc: str, tags: str) -> str:
    return PROMPT_TEMPLATE.format(doc=doc, tags=tags)


def create_binding(
    api_key: str = API_KEY,
    api_base: str = API_BASE,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
) -> RunnableBinding:
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model_name=model_name,
        temperature=temperature,
    )
    return llm.with_structured_output(Sentiment)


def create_output(structured_llm: RunnableBinding, prompt: str) -> dict[str, str]:
    try:
        res = structured_llm.invoke(prompt)
        sentiment = res.sentiment[0].value
        return dict(sentiment=sentiment, reason=np.nan)
    except Exception as e:
        print(e)
        return np.nan


# -----------------------------
# Load test set
# -----------------------------
df = pd.read_json(TEST_JSON_PATH, lines=True)  # Import testing set

# -----------------------------
# Inference
# -----------------------------
print("Starting LLM inference...")
structured_llm = create_binding()
tqdm.pandas()

start_time = time.time()

df["AI"] = df.progress_apply(
    lambda x: create_output(structured_llm, create_prompt(x["Text"], x["TICKER"])),
    axis=1,
)

end_time = time.time()
elapsed = end_time - start_time

# -----------------------------
# Save timing info
# -----------------------------
with open(SAVE_TIME_PATH, "w") as f:
    f.write(
        f"Total Inference Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n\n"
    )

# -----------------------------
# Extract output fields
# -----------------------------
df[["AI_sentiment", "AI_reason"]] = df["AI"].apply(
    lambda x: (
        pd.Series(
            {
                "AI_sentiment": (
                    x["sentiment"].capitalize()
                    if isinstance(x, dict) and pd.notna(x["sentiment"])
                    else np.nan
                ),
                "AI_reason": (
                    x["reason"]
                    if isinstance(x, dict) and pd.notna(x["reason"])
                    else np.nan
                ),
            }
        )
        if isinstance(x, dict)
        else pd.Series({"AI_sentiment": np.nan, "AI_reason": np.nan})
    )
)

# -----------------------------
# Save output file
# -----------------------------
df.to_json(SAVE_JSON_PATH, orient="records", lines=True)

print("LLM inference completed and results saved.")
