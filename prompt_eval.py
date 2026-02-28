import dspy
import json
import os
import re
import warnings

# Suppress Pydantic serialization warnings (non-critical)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ============================================================================
# 1. Define Signatures and Modules (Must match training scripts exactly)
# ============================================================================

# --- LLM1: Relevance Check ---
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
        # Use dspy.Predict instead of ChainOfThought to avoid reasoning output
        self.relevance_checker = dspy.Predict(RelevanceCheck)
    
    def forward(self, query):
        result = self.relevance_checker(query=query)            
        return result.output

# --- LLM2: Paragraph Analysis ---
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
        # Use dspy.Predict instead of ChainOfThought to avoid reasoning output
        self.analyzer = dspy.Predict(AnalyzeParagraph)
    
    def forward(self, paragraph, target_company):
        result = self.analyzer(paragraph=paragraph, target_company=target_company)
        return result

# --- LLM3: Aggregation ---
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
        # Use dspy.Predict instead of ChainOfThought to avoid reasoning output
        self.aggregator = dspy.Predict(AggregateResults)
    
    def forward(self, paragraph_analyses):
        return self.aggregator(paragraph_analyses=paragraph_analyses)

# ============================================================================
# 2. Load Models and Configure DSPy
# ============================================================================

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️  Warning: OPENAI_API_KEY not set.")

lm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key, temperature=0.2)
dspy.configure(lm=lm)

print("📂 Loading optimized models...")

try:
    llm1 = RelevanceModule()
    llm1.load("llm1_optimized.json")
    print("✅ LLM1 loaded")
except Exception as e:
    print(f"⚠️  LLM1 load failed: {e}")

try:
    llm2 = ParagraphAnalysisModule()
    llm2.load("llm2_optimized_3.json")
    print("✅ LLM2 loaded")
except Exception as e:
    print(f"⚠️  LLM2 load failed: {e}")

try:
    llm3 = AggregationModule()
    llm3.load("llm3_optimized.json")
    print("✅ LLM3 loaded")
except Exception as e:
    print(f"⚠️  LLM3 load failed: {e}")

# ============================================================================
# 3. Evaluation Loop
# ============================================================================

print("\n🚀 Starting Whole Flow Evaluation...")

with open('whole_flow_examples.json', 'r') as f:
    examples = json.load(f)

correct_count = 0
total_count = len(examples)

for i, example in enumerate(examples, 1):
    print(f"\n" + "="*60)
    print(f"🧪 Test Case {i}/{total_count}")
    print("="*60)

    # --- Synthesize Query ---
    target_company_str = example['target_company']
    if "The target company is" in target_company_str:
        company = target_company_str.replace("The target company is", "").strip(" .")
        query = f"Is {company} present in the agreement?"
    else:
        query = "Is there a target company in the agreement?"
    
    print(f"📝 Generated Query: {query}")

    # --- Step 1: LLM1 ---
    print("\n[Step 1] Running LLM1 (Relevance)...")
    try:
        output1 = llm1(query)
        print(f"   Output: {output1}")
    except Exception as e:
        print(f"   ❌ Error in LLM1: {e}")
        continue

    # Check for irrelevance
    start_tag = "<user_message>"
    end_tag = "</user_message>"
    if start_tag in output1 and end_tag in output1:
        print("   Verdict: Irrelevant Query (Stopped)")
        # In this specific test set, we expect relevance, so this might be a failure if expected output exists
        # But following the flow, we stop here.
        continue

    # --- Step 2: LLM2 ---
    print("\n[Step 2] Running LLM2 (Paragraph Analysis)...")
    docs = example['documents']
    paragraph_analyses = []
    
    for j, doc in enumerate(docs, 1):
        try:
            pred2 = llm2(paragraph=doc, target_company=output1)
            
            # Get JSON output from prediction
            json_output = getattr(pred2, 'json_output', '{}')
            print(f"   Paragraph {j} Output: {json_output}")
            
            # Parse the JSON to reformat it for LLM3 (which expects the old format)
            try:
                parsed = json.loads(json_output)
                # Reformat to match LLM3's expected input format
                formatted = json.dumps({
                    "Buyer": parsed.get("Buyer", "Not stated"),
                    "Buyer Representative": parsed.get("Buyer Representative", "Not stated"),
                    "Seller": parsed.get("Seller", "Not stated"),
                    "Seller Representative": parsed.get("Seller Representative", "Not stated"),
                    "Third-Party Representation": parsed.get("Third-Party Representation", "Not stated"),
                    "Target Company Mentioned": parsed.get("Target Company Mentioned", "No")
                })
                paragraph_analyses.append(formatted)
            except json.JSONDecodeError:
                # If parsing fails, use the raw output
                paragraph_analyses.append(json_output)
            
        except Exception as e:
            print(f"   ❌ Error in LLM2 (Para {j}): {e}")
            paragraph_analyses.append("{}")

    # --- Step 3: LLM3 ---
    print("\n[Step 3] Running LLM3 (Aggregation)...")
    try:
        # Concatenate analyses with newlines, matching the original app logic
        formatted_analyses = "\n".join(paragraph_analyses)
        pred3 = llm3(paragraph_analyses=formatted_analyses)
        output3_json_str = getattr(pred3, 'json_output', '{}')
        print(f"   Raw Output: {output3_json_str}")
    except Exception as e:
        print(f"   ❌ Error in LLM3: {e}")
        continue

    # --- Grading ---
    print("\n[Grading]")
    expected_json = example['expected_output_3']
    
    try:
        # Robust JSON extraction
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output3_json_str, re.DOTALL)
        if json_match:
            output3_json_str = json_match.group(0)
            
        parsed_answer = json.loads(output3_json_str)
    except json.JSONDecodeError as e:
        print(f"   ❌ Incorrect (Invalid JSON: {e})")
        continue

    # Verify keys
    required_keys = ["buyer_firm", "seller_firm", "third_party", "contains_target_firm"]
    missing_keys = [key for key in required_keys if key not in parsed_answer]
    if missing_keys:
        print(f"   ❌ Incorrect (Missing Keys: {', '.join(missing_keys)})")
        continue

    # Compare values
    incorrect_values = []
    for key in required_keys:
        val_pred = parsed_answer.get(key)
        val_exp = expected_json.get(key)
        
        # Normalize for comparison
        str_pred = str(val_pred).strip().lower()
        str_exp = str(val_exp).strip().lower()
        
        # Handle "Not stated" variations
        if str_exp == "not stated" and str_pred in ["not stated", "none", "n/a", ""]:
            continue
            
        if str_pred != str_exp:
            incorrect_values.append(key)

    if incorrect_values:
        print(f"   ❌ Incorrect (Values mismatch)")
        for key in required_keys:
            val_pred = parsed_answer.get(key)
            val_exp = expected_json.get(key)
            status = "❌" if key in incorrect_values else "✅"
            print(f"      {status} {key}: Expected '{val_exp}' | Got '{val_pred}'")
    else:
        print("   ✅ Correct")
        correct_count += 1

print("\n" + "="*60)
print(f"📊 Final Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")
print("="*60)
