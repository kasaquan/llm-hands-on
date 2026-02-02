import json
import os
import re
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2

# Verbose logging flag - set to True for detailed output
VERBOSE = False

# 1. Load Prompts
def load_prompt(filename):
    with open(filename, 'r') as f:
        return f.read()

system_prompt_1 = load_prompt('llm1_prompt.txt')
system_prompt_2 = load_prompt('llm2_prompt_raw.txt')
system_prompt_3 = load_prompt('llm3_prompt.txt')

# 2. Helper Functions
def run_llm(system_prompt, user_message):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OpenAI API call: {str(e)}")
        return ""

def parse_llm2_output_to_json(text):
    """
    Parses the custom [[ ## field ## ]] format from LLM2 and converts it 
    to the JSON format expected by LLM3's few-shot examples.
    """
    field_map = {
        "buyer": "Buyer",
        "buyer_representative": "Buyer Representative",
        "seller": "Seller",
        "seller_representative": "Seller Representative",
        "third_party_representation": "Third-Party Representation",
        "target_company_mentioned": "Target Company Mentioned"
    }
    
    data = {}
    
    # Simple parsing logic
    # We look for [[ ## key ## ]]\nvalue
    pattern = r"\[\[ ## (.*?) ## \]\]\s*(.*?)(?=\s*\[\[|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    parsed_fields = {k: v.strip() for k, v in matches}
    
    for llm_key, json_key in field_map.items():
        data[json_key] = parsed_fields.get(llm_key, "Not stated")
        
    return json.dumps(data)

# 3. Load Data
with open('whole_flow_examples.json', 'r') as f:
    examples = json.load(f)

correct_count = 0
total_count = len(examples)
failed_cases = []  # Track failed test cases for summary

print("\n🚀 Starting Whole Flow Evaluation (Raw OpenAI API)...")
print(f"   Model: {MODEL_NAME}, Temperature: {TEMPERATURE}, Verbose: {VERBOSE}")

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

    # --- Step 1: LLM1 (Relevance) ---
    print("\n[Step 1] Running LLM1 (Relevance)...")
    output1 = run_llm(system_prompt_1, query)
    print(f"   Output: {output1}")

    # Check for irrelevance
    if "<user_message>Query is not relevant" in output1:
        print("   Verdict: Irrelevant Query (Stopped)")
        continue

    # --- Step 2: LLM2 (Paragraph Analysis) ---
    print("\n[Step 2] Running LLM2 (Paragraph Analysis)...")
    docs = example['documents']
    paragraph_analyses = []
    
    for j, doc in enumerate(docs, 1):
        # Format user message for LLM2
        user_msg_2 = f"Target company context:\n{output1}\n\nParagraph:\n{doc}"
        
        if VERBOSE:
            print(f"\n   --- Paragraph {j} ---")
            print(f"   📄 Input Paragraph ({len(doc)} chars):")
            print(f"   {doc[:300]}{'...' if len(doc) > 300 else ''}")
        
        pred2 = run_llm(system_prompt_2, user_msg_2)
        
        if VERBOSE:
            print(f"   🤖 Raw LLM2 Output:")
            print(f"   {pred2}")
        else:
            print(f"   Paragraph {j} Output: {pred2[:100]}...") # Truncate for display
        
        # Check if output is already JSON or needs parsing
        try:
            # Try to parse as JSON directly first
            json.loads(pred2)
            json_analysis = pred2
            if VERBOSE:
                print(f"   ✅ Output is valid JSON")
        except json.JSONDecodeError:
            # Fall back to parsing [[ ## field ## ]] format
            json_analysis = parse_llm2_output_to_json(pred2)
            if VERBOSE:
                print(f"   ⚠️ Parsed from field format to JSON:")
                print(f"   {json_analysis}")
        
        paragraph_analyses.append(json_analysis)

    # --- Step 3: LLM3 (Aggregation) ---
    print("\n[Step 3] Running LLM3 (Aggregation)...")
    formatted_analyses = "\n".join(paragraph_analyses)
    user_msg_3 = f"Extracted information:\n{formatted_analyses}"
    
    if VERBOSE:
        print(f"   📥 Input to LLM3:")
        for idx, analysis in enumerate(paragraph_analyses, 1):
            print(f"      [{idx}] {analysis}")
    
    output3 = run_llm(system_prompt_3, user_msg_3)
    
    if VERBOSE:
        print(f"   🤖 Raw LLM3 Output:")
        print(f"   {output3}")
    
    # Extract JSON output from LLM3 response
    # LLM3 output format: [[ ## json_output ## ]]\n{...}\n[[ ## completed ## ]]
    match = re.search(r"\[\[ ## json_output ## \]\]\s*(.*?)(?=\s*\[\[|$)", output3, re.DOTALL)
    if match:
        output3_json_str = match.group(1).strip()
        if VERBOSE:
            print(f"   ℹ️ Extracted from [[ ## json_output ## ]] format")
    else:
        # Fallback: try to find first JSON object
        output3_json_str = output3
    
    print(f"   Final Output: {output3_json_str}")

    # --- Grading ---
    print("\n[Grading]")
    expected_json = example['expected_output_3']
    
    if VERBOSE:
        print(f"   📋 Expected Output: {json.dumps(expected_json)}")
    
    try:
        # Robust JSON extraction
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output3_json_str, re.DOTALL)
        if json_match:
            output3_json_str = json_match.group(0)
            
        parsed_answer = json.loads(output3_json_str)
    except json.JSONDecodeError as e:
        print(f"   ❌ Incorrect (Invalid JSON: {e})")
        failed_cases.append({
            "case": i,
            "query": query,
            "target": target_company_str,
            "incorrect_keys": ["JSON_PARSE_ERROR"],
            "expected": expected_json,
            "got": {"error": str(e), "raw": output3_json_str[:200]}
        })
        continue

    # Verify keys
    required_keys = ["buyer_firm", "seller_firm", "third_party", "contains_target_firm"]
    missing_keys = [key for key in required_keys if key not in parsed_answer]
    if missing_keys:
        print(f"   ❌ Incorrect (Missing Keys: {', '.join(missing_keys)})")
        failed_cases.append({
            "case": i,
            "query": query,
            "target": target_company_str,
            "incorrect_keys": ["MISSING_KEYS: " + ", ".join(missing_keys)],
            "expected": expected_json,
            "got": parsed_answer
        })
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
        failed_cases.append({
            "case": i,
            "query": query,
            "target": target_company_str,
            "incorrect_keys": incorrect_values,
            "expected": expected_json,
            "got": parsed_answer
        })
    else:
        print("   ✅ Correct")
        correct_count += 1

print("\n" + "="*60)
print(f"📊 Final Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")
print("="*60)

if failed_cases and VERBOSE:
    print("\n" + "="*60)
    print("📋 FAILED CASES SUMMARY")
    print("="*60)
    for fc in failed_cases:
        print(f"\n❌ Case {fc['case']}: {fc['target']}")
        print(f"   Query: {fc['query']}")
        print(f"   Incorrect fields: {', '.join(fc['incorrect_keys'])}")
        print(f"   Expected: {json.dumps(fc['expected'])}")
        print(f"   Got:      {json.dumps(fc['got'])}")
