import dspy
import json
import os
import random
import re
import warnings

# Suppress Pydantic serialization warnings (non-critical)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load training examples
with open('llm2_examples.json', 'r') as f:
    examples_data = json.load(f)

# Shuffle for random train/test split
random.shuffle(examples_data)

# Split into 70% training and 30% testing
total_examples = len(examples_data)
train_size = int(total_examples * 0.7)
train_data = examples_data[:train_size]
test_data = examples_data[train_size:]

print(f"📊 Dataset split:")
print(f"   Total examples: {total_examples}")
print(f"   Training (70%): {len(train_data)} examples")
print(f"   Testing (30%): {len(test_data)} examples\n")

# Configure DSPy with OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️  Warning: OPENAI_API_KEY not set. Please set it before running.")
    print("   export OPENAI_API_KEY='your-key-here'")

# Set up student (production) and teacher (optimization) models
student_llm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key, temperature=0.2)
teacher_llm = dspy.LM(model="openai/gpt-4o", api_key=openai_api_key)

class AnalyzeParagraph(dspy.Signature):
    """
    Analyzes a paragraph from a legal file to extract required specific entities relative to the target company.
    1. Buyer's representative law firm
    2. Seller's representative law firm
    3. Any third-party law firm present
    4. Whether the target company is mentioned in the paragraph

    If an entity is not mentioned, use 'Not stated'.
    If an entity does not have a name, use 'Not stated', don't use 'the Company', 'the Purchaser', 'the Investor', etc.
    Output ONLY valid JSON with no additional text.
    """
    paragraph = dspy.InputField(desc="One paragraph from the contract")
    target_company = dspy.InputField(desc="The identified target company from Step 1")
    
    json_output = dspy.OutputField(desc='A valid JSON object with exactly these keys: {"Buyer": "string", "Buyer Representative": "string", "Seller": "string", "Seller Representative": "string", "Third-Party Representation": "string", "Target Company Mentioned": "Yes or No"}. Output ONLY the JSON, no other text.')


# Helper to create JSON output string from answer dict
def create_json_output(answer):
    return json.dumps({
        "Buyer": answer.get('Buyer', 'Not stated'),
        "Buyer Representative": answer.get('Buyer Representative', 'Not stated'),
        "Seller": answer.get('Seller', 'Not stated'),
        "Seller Representative": answer.get('Seller Representative', 'Not stated'),
        "Third-Party Representation": answer.get('Third-Party Representation', 'Not stated'),
        "Target Company Mentioned": answer.get('Target Company Mentioned', 'No')
    })

# Create training examples with single JSON output field
training_examples = []
for item in train_data:
    answer = item['answer']
    training_examples.append(
        dspy.Example(
            paragraph=item['paragraph'],
            target_company=item['target_company'],
            json_output=create_json_output(answer)
        ).with_inputs("paragraph", "target_company")
    )

# Create validation examples (for MIPROv2) - must be dspy.Example objects
val_examples = []
for item in test_data[:10] if len(test_data) > 10 else test_data:  # Use subset for validation
    answer = item['answer']
    val_examples.append(
        dspy.Example(
            paragraph=item['paragraph'],
            target_company=item['target_company'],
            json_output=create_json_output(answer)
        ).with_inputs("paragraph", "target_company")
    )

# Create test examples (for final evaluation) - keep as dicts for easier testing
test_examples = []
for item in test_data:
    answer = item['answer']
    test_examples.append({
        'paragraph': item['paragraph'],
        'target_company': item['target_company'],
        'expected_output': {
            'Buyer': answer.get('Buyer', 'Not stated'),
            'Buyer Representative': answer.get('Buyer Representative', 'Not stated'),
            'Seller': answer.get('Seller', 'Not stated'),
            'Seller Representative': answer.get('Seller Representative', 'Not stated'),
            'Third-Party Representation': answer.get('Third-Party Representation', 'Not stated'),
            'Target Company Mentioned': answer.get('Target Company Mentioned', 'No')
        }
    })

# Create the module
class ParagraphAnalysisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use dspy.Predict instead of ChainOfThought to avoid reasoning output
        # The online eval system expects direct JSON output without reasoning
        self.analyzer = dspy.Predict(AnalyzeParagraph)
    
    def forward(self, paragraph, target_company):
        result = self.analyzer(paragraph=paragraph, target_company=target_company)
        return result

print("🔨 Training LLM2 with MIPROv2 optimizer...")
print("📚 Teacher (for optimization): GPT-4o")
print("📚 Student (production): GPT-4o-mini\n")

# Create the module
print("🎓 Creating ParagraphAnalysisModule...")
module = ParagraphAnalysisModule()

# Validation metric
def validate_output(example, pred, trace=None):
    """
    MIPROv2 metric function: (example, pred, trace=None) -> float
    
    Returns a normalized score from 0.0 to 1.0 based on how many fields are correct.
    Each correct field = 1/6 points, normalized to 0.0-1.0 range.
    """
    fields = ['Buyer', 'Buyer Representative', 'Seller', 'Seller Representative', 
              'Third-Party Representation', 'Target Company Mentioned']
    
    max_score = len(fields)
    score = 0.0
    
    # Parse expected JSON from example
    try:
        expected_json = json.loads(getattr(example, 'json_output', '{}'))
    except json.JSONDecodeError:
        return 0.0
    
    # Parse predicted JSON
    try:
        pred_json_str = getattr(pred, 'json_output', '{}')
        # Try to extract JSON if mixed with other text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', pred_json_str, re.DOTALL)
        if json_match:
            pred_json_str = json_match.group(0)
        predicted_json = json.loads(pred_json_str)
    except json.JSONDecodeError:
        return 0.0
    
    for field in fields:
        expected_val = str(expected_json.get(field, "")).strip().lower()
        predicted_val = str(predicted_json.get(field, "")).strip().lower()
        
        # Check if field is correct (case-insensitive matching)
        if expected_val == predicted_val:
            score += 1.0
        # Handle "Not stated" variations
        elif expected_val == "not stated" and predicted_val in ["not stated", "none", "n/a", ""]:
            score += 1.0
    
    # Normalize to 0.0-1.0 range
    return score / max_score if max_score > 0 else 0.0

# Use MIPROv2 optimizer for production with gpt-4o-mini
print("🏗️ Setting up MIPROv2 optimizer...")
compiled_model_path = "llm2_optimized.json"

if os.path.exists(compiled_model_path):
    print(f"📂 Loading optimized model from {compiled_model_path}...")
    module.load(compiled_model_path)
    optimized_program = module
    print("✅ Model loaded successfully!\n")
else:
    try:
        from dspy.teleprompt import MIPROv2
        
        # MIPROv2 with auto="light" for cost-effective optimization
        # You can use "medium" or "heavy" for better results but higher cost
        # prompt_model: uses teacher to generate instruction proposals
        # task_model: uses student to evaluate candidates (optimize for production model)
        # Note: When auto is set, num_candidates and num_trials are set automatically
        teleprompter = MIPROv2(
            metric=validate_output,
            prompt_model=teacher_llm,  # Teacher generates instruction proposals
            task_model=student_llm,     # Student evaluates candidates (optimize for production)
            auto="medium",  # Options: "light", "medium", "heavy" (auto sets num_candidates)
            init_temperature=1.0
        )
        
        print("🔄 Compiling module with MIPROv2...")
        print("   (This may take a few minutes and will use API calls)\n")
        
        optimized_program = teleprompter.compile(
            student=module,
            trainset=training_examples,
            valset=val_examples  # Validation set as dspy.Example objects
        )
        
        print("✅ Module optimized successfully with MIPROv2!\n")
        
    except ImportError:
        print("⚠️  MIPROv2 not available. Trying BootstrapFewShot as fallback...")
        from dspy.teleprompt import BootstrapFewShot
        teleprompter = BootstrapFewShot(metric=validate_output, teacher=teacher_llm)
        optimized_program = teleprompter.compile(student=module, trainset=training_examples)
        print("✅ Module optimized with BootstrapFewShot!\n")
    except Exception as e:
        print(f"⚠️  Error with MIPROv2: {e}")
        print("   Falling back to basic module...")
        optimized_program = module
    
    # Save the optimized program
    print(f"💾 Saving optimized model to {compiled_model_path}...")
    optimized_program.save(compiled_model_path)
    print("✅ Model saved!\n")

# Test the optimized module on the test set
print("🧪 Testing Optimized Module (GPT-4o-mini) on test set:")
print("=" * 60)
dspy.configure(lm=student_llm)  # Use student LM for inference

correct = 0
total = len(test_examples)

for i, test_item in enumerate(test_examples, 1):
    paragraph = test_item['paragraph']
    target_company = test_item['target_company']
    expected = test_item['expected_output']
    
    try:
        result = optimized_program(paragraph=paragraph, target_company=target_company)
        
        # Parse predicted JSON
        pred_json_str = getattr(result, 'json_output', '{}')
        # Try to extract JSON if mixed with other text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', pred_json_str, re.DOTALL)
        if json_match:
            pred_json_str = json_match.group(0)
        
        try:
            predicted_json = json.loads(pred_json_str)
        except json.JSONDecodeError:
            predicted_json = {}
        
        # Check if all fields match
        fields = ['Buyer', 'Buyer Representative', 'Seller', 'Seller Representative', 
                  'Third-Party Representation', 'Target Company Mentioned']
        all_match = True
        
        for field in fields:
            expected_val = expected[field].strip()
            predicted_val = str(predicted_json.get(field, "")).strip()
            
            # Flexible matching for "Not stated"
            if expected_val == "Not stated":
                if predicted_val.lower() not in ["not stated", "none", "n/a", ""]:
                    all_match = False
            elif predicted_val.lower() != expected_val.lower():
                all_match = False
        
        if all_match:
            correct += 1
        
        status = "✅" if all_match else "❌"
        print(f"\n{status} Test {i}/{total}")
        print(f"   Target: {target_company[:50]}...")
        print(f"   Paragraph: {paragraph[:200]}{'...' if len(paragraph) > 200 else ''}")
        for field in fields:
            expected_val = expected[field]
            predicted_val = predicted_json.get(field, 'N/A')
            match_indicator = "✓" if str(expected_val).strip().lower() == str(predicted_val).strip().lower() else "✗"
            print(f"   {match_indicator} {field}: Expected '{expected_val}' | Got '{predicted_val}'")
        
    except Exception as e:
        print(f"\n❌ Error with test {i}: {e}")

print("\n" + "=" * 60)
print(f"📊 Test Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("=" * 60)

# Extract the complete production prompt using named_predictors()
print("\n" + "=" * 80)
print("📋 EXTRACTING COMPLETE PRODUCTION PROMPT")
print("=" * 80)

# 1. Print the instructions and demos from the optimized predictor
for name, pred in optimized_program.named_predictors():
    print(f"\nPredictor: {name}")
    print("-" * 40)
    print(f"Instructions: {pred.signature.instructions}")
    print(f"Number of Few-Shot Examples (Demos): {len(pred.demos)}")
    print("-" * 40)
    
# 2. Inspect the history of the student LLM to see the EXACT full prompt sent
print("\n👀 VIEWING LAST ACTUAL LLM PROMPT (What was sent to the API)")
print("=" * 80)
student_llm.inspect_history(n=1)
print("=" * 80)

