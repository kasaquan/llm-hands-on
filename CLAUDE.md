# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='your-key-here'
```

## Running Scripts

```bash
# Optimize prompts for each LLM stage
python train_llm1_dspy.py   # Trains relevance checker (saves llm1_optimized.json)
python train_llm2_dspy.py   # Trains paragraph analyzer (saves llm2_optimized.json)
python train_llm3_dspy.py   # Trains aggregator (saves llm3_optimized.json)

# Evaluate the full pipeline
python prompt_eval.py       # End-to-end evaluation using DSPy optimized models
python example.py           # End-to-end evaluation using raw OpenAI API (baseline)
```

Training scripts skip optimization and load from the cached `*_optimized.json` file if it exists. Delete the JSON file to force re-optimization.

## Architecture

This project uses DSPy to optimize prompts for a 3-stage legal contract analysis pipeline. The pipeline answers queries like "Is [Company X] mentioned in this agreement?" by processing contract documents.

**Pipeline stages:**

1. **LLM1 - Relevance Check** (`train_llm1_dspy.py`, `RelevanceCheck` signature)
   - Input: user query
   - Output: `"The target company is [Name]."` or `<user_message>Query is not relevant to the intended task.</user_message>`
   - If irrelevant, the pipeline stops early

2. **LLM2 - Paragraph Analysis** (`train_llm2_dspy.py`, `AnalyzeParagraph` signature)
   - Input: one contract paragraph + target company name from LLM1
   - Output: JSON with `{Buyer, Buyer Representative, Seller, Seller Representative, Third-Party Representation, Target Company Mentioned}`
   - Called once per document paragraph; results are concatenated

3. **LLM3 - Aggregation** (`train_llm3_dspy.py`, `AggregateResults` signature)
   - Input: concatenated JSON strings from all LLM2 calls
   - Output: final JSON with `{buyer_firm, seller_firm, third_party, contains_target_firm}`

**DSPy optimization pattern (Architect-Intern):**
- Student (production): `gpt-4o-mini` at temperature 0.2
- Teacher (optimization): `gpt-4o` or `gpt-4o` (LLM1 uses `gpt-5.2`)
- LLM1 & LLM2 use `MIPROv2` with `auto="medium"`
- LLM3 uses `BootstrapFewShot`
- All modules use `dspy.Predict` (not `ChainOfThought`) to avoid reasoning output in responses

**Key data files:**
- `user_queries.json` - LLM1 training data (100 queries with expected outputs)
- `llm2_examples.json` - LLM2 training data (paragraph + target_company + answer)
- `llm3_examples.json` - LLM3 training data (paragraph analyses inputs + aggregated answer)
- `whole_flow_examples.json` - End-to-end test cases with all three expected outputs
- `llm1_prompt.txt`, `llm2_prompt_raw.txt`, `llm3_prompt.txt` - Manually crafted baseline prompts

**Evaluation (`prompt_eval.py`):**
- Loads all three optimized modules from their respective JSON files
- Runs each test case through the full pipeline
- Grades final JSON output against `expected_output_3` from `whole_flow_examples.json`
- Required output keys: `buyer_firm`, `seller_firm`, `third_party`, `contains_target_firm`

**Baseline (`example.py`):**
- Uses raw OpenAI API (no DSPy) with manually crafted prompts from `.txt` files
- Useful for comparing baseline performance against DSPy-optimized versions
