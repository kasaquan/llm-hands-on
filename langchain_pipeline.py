"""
LangChain pipeline that chains the 3 DSPy-optimized LLM stages for
complete legal contract analysis.

Pipeline:
  Input: {"query": str, "documents": [str]}
    │
    ▼
  [Stage 1] LLM1 - Relevance Check
    - Extracts the target company from the user query
    - Returns early with a user_message if query is irrelevant
    │
    ▼
  [Stage 2] LLM2 - Paragraph Analysis (maps over each document)
    - Analyzes each paragraph against the target company
    - Produces per-paragraph JSON with Buyer/Seller/Representative/etc.
    │
    ▼
  [Stage 3] LLM3 - Aggregation
    - Consolidates all paragraph analyses into a final JSON result
    │
    ▼
  Output: {"result": {...}, "is_irrelevant": bool, "target_company": str}

Usage:
  python langchain_pipeline.py
"""

import dspy
import json
import os
import re
import warnings

from langchain_core.runnables import RunnableLambda

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ============================================================================
# 1. DSPy Signatures and Modules (must match training scripts exactly)
# ============================================================================

class RelevanceCheck(dspy.Signature):
    """Determines if the user's query mentions any target company.

    1. If a target company is mentioned, output: 'The target company is [Company Name].'
    2. If no target company is found, you MUST output EXACTLY:
    <user_message>Query is not relevant to the intended task.</user_message>

    IMPORTANT: Do not miss the <user_message> tags in the negative case.
    """
    query = dspy.InputField(desc="The user's query about a contract")
    output = dspy.OutputField(desc="The final answer. Ensure exact XML format for negative cases.")


class RelevanceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.relevance_checker = dspy.Predict(RelevanceCheck)

    def forward(self, query):
        result = self.relevance_checker(query=query)
        return result.output


class AnalyzeParagraph(dspy.Signature):
    """
    Analyzes a paragraph from a legal file to extract required specific entities relative to the target company.
    If an entity is not mentioned, use 'Not stated'.
    If an entity does not have a name, use 'Not stated', don't use 'the Company', 'the Purchaser', 'the Investor', etc.
    Output ONLY valid JSON with no additional text.
    """
    paragraph = dspy.InputField(desc="One paragraph from the contract")
    target_company = dspy.InputField(desc="The identified target company from Step 1")
    json_output = dspy.OutputField(desc='A valid JSON object with exactly these keys: {"Buyer": "string", "Buyer Representative": "string", "Seller": "string", "Seller Representative": "string", "Third-Party Representation": "string", "Target Company Mentioned": "Yes or No"}. Output ONLY the JSON, no other text.')


class ParagraphAnalysisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(AnalyzeParagraph)

    def forward(self, paragraph, target_company):
        return self.analyzer(paragraph=paragraph, target_company=target_company)


class AggregateResults(dspy.Signature):
    """
    Aggregates analysis from multiple paragraphs into a final structured JSON object.
    Consolidates law firm information and determines if target company appears in any paragraph.
    The goal is to identify the representative law firms of involved parties and determine if the target company is mentioned, ensuring the results are structured and accurate.
    """
    paragraph_analyses = dspy.InputField(desc="Concatenated strings of several paragraph analyses.")
    json_output = dspy.OutputField(desc="A valid JSON string with exactly 4 fields: buyer_firm (string), seller_firm (string), third_party (string), contains_target_firm (boolean). Output ONLY the JSON string, no other text.")


class AggregationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = dspy.Predict(AggregateResults)

    def forward(self, paragraph_analyses):
        return self.aggregator(paragraph_analyses=paragraph_analyses)


# ============================================================================
# 2. Load DSPy Models
# ============================================================================

def load_dspy_modules():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key-here'")

    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key, temperature=0.2)
    dspy.configure(lm=lm)

    print("Loading optimized DSPy modules...")

    llm1 = RelevanceModule()
    llm1.load("llm1_optimized.json")
    print("  LLM1 loaded (Relevance Check)")

    llm2 = ParagraphAnalysisModule()
    llm2.load("llm2_optimized.json")
    print("  LLM2 loaded (Paragraph Analysis)")

    llm3 = AggregationModule()
    llm3.load("llm3_optimized.json")
    print("  LLM3 loaded (Aggregation)")

    return llm1, llm2, llm3


# ============================================================================
# 3. LangChain Stage Functions
# ============================================================================

def _extract_json(raw: str) -> dict:
    """Extract and parse the first JSON object found in a string."""
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    return json.loads(raw)


def make_stage1(llm1_module: RelevanceModule):
    """Return a Runnable for Stage 1: Relevance Check."""

    def run(state: dict) -> dict:
        query = state["query"]
        output = llm1_module(query)

        if "<user_message>" in output and "</user_message>" in output:
            # Irrelevant query — extract the message and signal early exit
            msg = re.search(r'<user_message>(.*?)</user_message>', output, re.DOTALL)
            user_message = msg.group(1).strip() if msg else output
            return {**state, "target_company": None, "early_exit": user_message}

        return {**state, "target_company": output.strip(), "early_exit": None}

    return RunnableLambda(run)


def make_stage2(llm2_module: ParagraphAnalysisModule):
    """Return a Runnable for Stage 2: Paragraph Analysis (maps over documents)."""

    def run(state: dict) -> dict:
        if state.get("early_exit"):
            return state  # Pass through on early exit

        target_company = state["target_company"]
        documents = state["documents"]
        paragraph_analyses = []

        for doc in documents:
            pred = llm2_module(paragraph=doc, target_company=target_company)
            raw = getattr(pred, "json_output", "{}")

            try:
                parsed = _extract_json(raw)
                formatted = json.dumps({
                    "Buyer": parsed.get("Buyer", "Not stated"),
                    "Buyer Representative": parsed.get("Buyer Representative", "Not stated"),
                    "Seller": parsed.get("Seller", "Not stated"),
                    "Seller Representative": parsed.get("Seller Representative", "Not stated"),
                    "Third-Party Representation": parsed.get("Third-Party Representation", "Not stated"),
                    "Target Company Mentioned": parsed.get("Target Company Mentioned", "No"),
                })
            except (json.JSONDecodeError, ValueError):
                formatted = raw  # Fallback: pass raw text

            paragraph_analyses.append(formatted)

        return {**state, "paragraph_analyses": "\n".join(paragraph_analyses)}

    return RunnableLambda(run)


def make_stage3(llm3_module: AggregationModule):
    """Return a Runnable for Stage 3: Aggregation."""

    def run(state: dict) -> dict:
        if state.get("early_exit"):
            return {
                "result": state["early_exit"],
                "is_irrelevant": True,
                "target_company": None,
            }

        pred = llm3_module(paragraph_analyses=state["paragraph_analyses"])
        raw = getattr(pred, "json_output", "{}")

        try:
            result = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            result = {"error": "Failed to parse LLM3 output", "raw": raw}

        return {
            "result": result,
            "is_irrelevant": False,
            "target_company": state["target_company"],
        }

    return RunnableLambda(run)


# ============================================================================
# 4. Build the Pipeline
# ============================================================================

def build_pipeline(llm1, llm2, llm3):
    """Chain the 3 stages into a single LangChain LCEL pipeline."""
    return make_stage1(llm1) | make_stage2(llm2) | make_stage3(llm3)


# ============================================================================
# 5. Main — Demo Run
# ============================================================================

if __name__ == "__main__":
    llm1, llm2, llm3 = load_dspy_modules()
    pipeline = build_pipeline(llm1, llm2, llm3)

    print("\n" + "=" * 60)
    print("LangChain 3-Stage Contract Analysis Pipeline")
    print("=" * 60)

    # Load a few whole-flow examples for demo
    with open("whole_flow_examples.json", "r") as f:
        examples = json.load(f)

    correct = 0
    total = len(examples)

    for i, example in enumerate(examples, 1):
        print(f"\n--- Test {i}/{total} ---")

        query = example["user_query"]
        print(f"Query: {query}")

        output = pipeline.invoke({"query": query, "documents": example["documents"]})

        if output["is_irrelevant"]:
            print(f"Result: IRRELEVANT — {output['result']}")
            if example.get("expected_is_irrelevant"):
                print("PASS")
                correct += 1
            else:
                print("FAIL: expected a structured result but got irrelevant")
            continue

        print(f"Target Company: {output['target_company']}")
        print(f"Final Result: {json.dumps(output['result'], indent=2)}")

        # Grade against expected
        expected = example["expected_output_3"]
        required_keys = ["buyer_firm", "seller_firm", "third_party", "contains_target_firm"]
        result = output["result"]

        mismatches = []
        for key in required_keys:
            pred_val = str(result.get(key, "")).strip().lower()
            exp_val = str(expected.get(key, "")).strip().lower()
            if exp_val == "not stated" and pred_val in ["not stated", "none", "n/a", ""]:
                continue
            if pred_val != exp_val:
                mismatches.append(f"{key}: expected '{expected[key]}' got '{result.get(key)}'")

        if mismatches:
            print(f"FAIL: {'; '.join(mismatches)}")
        else:
            print("PASS")
            correct += 1

    print("\n" + "=" * 60)
    print(f"Score: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 60)
