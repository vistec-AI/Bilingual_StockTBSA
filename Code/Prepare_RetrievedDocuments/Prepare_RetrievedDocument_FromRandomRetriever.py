# -----------------------------
# Prepare Randomly Selected Few-Shot Examples for Financial TBSA
# -----------------------------
"""
Randomly select few-shot examples from an ICL data pool
and combine them with the test dataset for LLM inference.
"""

# -----------------------------
# Imports Libraries
# -----------------------------
import random
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
ICL_DATA_PATH = "./path/to/icl_data_pool.json"  # Train-validation data pool used for random selection
DATASET_PATH = "./path/to/test_dataset.json"  # Thai or English test dataset
OUTPUT_PATH = "./path/to/random_retrieved_data.csv"  # Output dataset with randomly selected examples

NUM_SELECTED_EXAMPLES = 3  # Number of examples used for k-shot ICL
RANDOM_SEED = 42  # Random seed for reproducibility


# -----------------------------
# Few-Shot Example Template
# -----------------------------
FEW_SHOT_TEMPLATE = """EXAMPLE: {text}
TICKER: {ticker}
SENTIMENT_CLASS: {sentiment_class}
"""


# -----------------------------
# Randomly Select Few-Shot Examples
# -----------------------------
def add_random_examples_column(
    test_data: pd.DataFrame,
    icl_data: pd.DataFrame,
    n: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    random_examples_list = []

    for _, _ in test_data.iterrows():
        sampled_rows = icl_data.sample(
            n=n,
            random_state=random.randint(0, 10000),
        )

        examples = [
            {
                "text": example_row["Text"],
                "ticker": example_row["TICKER"],
                "sentiment_class": example_row["Sentiment_class"],
            }
            for _, example_row in sampled_rows.iterrows()
        ]

        random_examples_list.append(examples)

    test_data = test_data.copy()
    test_data["Random_example"] = random_examples_list

    return test_data


# -----------------------------
# Format Few-Shot Examples
# -----------------------------
def format_few_shot(examples: list[dict]) -> str:
    return "\n".join(FEW_SHOT_TEMPLATE.format(**example) for example in examples)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load the test dataset and ICL data pool
    test_data = pd.read_json(DATASET_PATH, lines=True)
    icl_data = pd.read_json(ICL_DATA_PATH, lines=True)

    # Randomly select examples for each test instance
    test_data_with_examples = add_random_examples_column(
        test_data,
        icl_data,
        n=NUM_SELECTED_EXAMPLES,
        seed=RANDOM_SEED,
    )

    # Format the selected examples
    test_data_with_examples["Few_shot_prompt"] = test_data_with_examples[
        "Random_example"
    ].apply(format_few_shot)

    # Create the target query
    test_data_with_examples["prompt"] = (
        "TARGET_ARTICLE: "
        + test_data_with_examples["Text"]
        + "\nTICKER: "
        + test_data_with_examples["TICKER"]
        + "\nSENTIMENT_CLASS: "
    )

    # This step is useful for later inference.
    test_data_with_examples["input_prompt"] = (
        test_data_with_examples["Few_shot_prompt"]
        + "\n"
        + test_data_with_examples["prompt"]
    )

    # Save the final dataset
    test_data_with_examples.to_csv(OUTPUT_PATH)

    print("Randomly selected few-shot examples saved successfully.")
