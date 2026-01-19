import dspy
import json
import os
import random
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
    If an entity is not mentioned, use 'Not stated'.
    """
    paragraph = dspy.InputField(desc="One paragraph from the contract")
    target_company = dspy.InputField(desc="The identified target company from Step 1")
    
    buyer = dspy.OutputField(desc="Name of the Buyer company")
    buyer_representative = dspy.OutputField(desc="Law firm or representative for the Buyer")
    
    seller = dspy.OutputField(desc="Name of the Seller company")
    seller_representative = dspy.OutputField(desc="Law firm or representative for the Seller")
    
    third_party_representation = dspy.OutputField(desc="Any third-party law firms or advisory roles mentioned")
    target_company_mentioned = dspy.OutputField(desc="Boolean: 'Yes' or 'No' indicating if target is explicitly mentioned")


class AggregateResults(dspy.Signature):
    """
    Aggregates analysis from multiple paragraphs into a final structured JSON object.
    Consolidates law firm information and determines if target company appears in any paragraph.
    """
    paragraph_analyses = dspy.InputField(desc="List of extraction results from all paragraphs")
    
    buyer_firm = dspy.OutputField(desc="Law firm representing the Buyer (consolidated from all paragraphs)")
    seller_firm = dspy.OutputField(desc="Law firm representing the Seller (consolidated from all paragraphs)")
    third_party = dspy.OutputField(desc="Third-party law firm mentioned (consolidated from all paragraphs)")
    contains_target_firm = dspy.OutputField(desc="Boolean: true if target company is mentioned in any paragraph, false otherwise")


# Create training examples
training_examples = []
for item in train_data:
    expected = item['expected_output']
    training_examples.append(
        dspy.Example(
            paragraph=item['paragraph'],
            target_company=item['target_company'],
            buyer=expected['buyer'],
            buyer_representative=expected['buyer_representative'],
            seller=expected['seller'],
            seller_representative=expected['seller_representative'],
            third_party_representation=expected['third_party_representation'],
            target_company_mentioned=expected['target_company_mentioned']
        ).with_inputs("paragraph", "target_company")
    )

# Create test examples
test_examples = []
for item in test_data:
    test_examples.append({
        'paragraph': item['paragraph'],
        'target_company': item['target_company'],
        'expected_output': item['expected_output']
    })

# Create the module
class ParagraphAnalysisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(AnalyzeParagraph)
    
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
    """Validate entity extraction results"""
    if not hasattr(pred, 'buyer'):
        return False
    
    # Check each field
    fields = ['buyer', 'buyer_representative', 'seller', 'seller_representative', 
              'third_party_representation', 'target_company_mentioned']
    
    all_correct = True
    for field in fields:
        expected = getattr(example, field, "").strip()
        predicted = getattr(pred, field, "").strip()
        
        # Allow some flexibility for "Not stated" variations
        if expected == "Not stated":
            if predicted.lower() in ["not stated", "none", "n/a", ""]:
                continue
        
        if predicted != expected:
            all_correct = False
            break
    
    return all_correct

# Use MIPROv2 optimizer for production with gpt-4o-mini
print("🏗️ Setting up MIPROv2 optimizer...")
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
        auto="light",  # Options: "light", "medium", "heavy" (auto sets num_candidates)
        init_temperature=1.0
    )
    
    print("🔄 Compiling module with MIPROv2...")
    print("   (This may take a few minutes and will use API calls)\n")
    
    optimized_program = teleprompter.compile(
        student=module,
        trainset=training_examples,
        valset=test_examples[:5] if len(test_examples) > 5 else test_examples  # Small validation set
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
        
        # Check if all fields match
        fields = ['buyer', 'buyer_representative', 'seller', 'seller_representative', 
                  'third_party_representation', 'target_company_mentioned']
        all_match = True
        
        for field in fields:
            expected_val = expected[field].strip()
            predicted_val = getattr(result, field, "").strip()
            
            # Flexible matching for "Not stated"
            if expected_val == "Not stated":
                if predicted_val.lower() not in ["not stated", "none", "n/a", ""]:
                    all_match = False
            elif predicted_val != expected_val:
                all_match = False
        
        if all_match:
            correct += 1
        
        status = "✅" if all_match else "❌"
        print(f"\n{status} Test {i}/{total}")
        print(f"   Target: {target_company[:50]}...")
        print(f"   Buyer: Expected '{expected['buyer']}' | Got '{getattr(result, 'buyer', 'N/A')}'")
        print(f"   Target mentioned: Expected '{expected['target_company_mentioned']}' | Got '{getattr(result, 'target_company_mentioned', 'N/A')}'")
        
    except Exception as e:
        print(f"\n❌ Error with test {i}: {e}")

print("\n" + "=" * 60)
print(f"📊 Test Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("=" * 60)

# Inspect the last prompt sent to the student model
print("\n🔍 Inspecting the last prompt sent to the Student LLM:")
print("=" * 60)
student_llm.inspect_history(n=1)
print("=" * 60)

