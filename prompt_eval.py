from openai import OpenAI
import json

# Initialize OpenAI (Make sure to set your API key)
client = OpenAI()

# --- PASTE YOUR DRAFT PROMPTS HERE ---
PROMPT_LLM1 = """Your LLM1 System Prompt here..."""
PROMPT_LLM2 = """Your LLM2 System Prompt here..."""
PROMPT_LLM3 = """Your LLM3 System Prompt here..."""

def run_chain(user_query, contract_text):
    print(f"--- Processing Query: {user_query} ---")
    
    # STEP 1: Check Relevance
    response1 = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.2,
        messages=[
            {"role": "system", "content": PROMPT_LLM1},
            {"role": "user", "content": user_query}
        ]
    )
    out1 = response1.choices[0].message.content
    print(f"[LLM1 Output]: {out1}")
    
    if "<user_message>" in out1:
        print("Query deemed irrelevant. Stopping.")
        return

    # STEP 2: Extract Info (Simulating paragraph split)
    # In reality, you'd split contract_text into paragraphs. Here we pass the whole chunk.
    response2 = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.2,
        messages=[
            {"role": "system", "content": PROMPT_LLM2},
            {"role": "user", "content": contract_text}
        ]
    )
    out2 = response2.choices[0].message.content
    print(f"[LLM2 Output]: {out2}")

    # STEP 3: Compile to JSON
    response3 = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.2,
        messages=[
            {"role": "system", "content": PROMPT_LLM3},
            {"role": "user", "content": out2} # Passing LLM2's output to LLM3
        ]
    )
    out3 = response3.choices[0].message.content
    print(f"[LLM3 Output]: {out3}")
    
    # Validation Check
    try:
        json_obj = json.loads(out3)
        print("✅ Valid JSON generated!")
        print(json.dumps(json_obj, indent=2))
    except:
        print("❌ Invalid JSON!")

# --- TEST DATA ---
sample_contract = """
This Agreement is entered into by and between Google LLC ("Buyer") and 
DeepMind Technologies ("Seller"). The law firm of Wachtell, Lipton, Rosen & Katz 
is representing the Seller.
"""

run_chain("Does this mention DeepMind?", sample_contract)