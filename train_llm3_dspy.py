import dspy
import json
import os
import random
import warnings

# Suppress Pydantic serialization warnings (non-critical)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load LLM3 training examples
with open('llm3_examples.json', 'r') as f:
    llm3_examples = json.load(f)

print(f"📊 Loaded {len(llm3_examples)} LLM3 training examples\n")

# Shuffle for random train/test split
random.shuffle(llm3_examples)

# Split into 70% training and 30% testing
total_examples = len(llm3_examples)
train_size = int(total_examples * 0.7)
train_data = llm3_examples[:train_size]
test_data = llm3_examples[train_size:]

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

class AggregateResults(dspy.Signature):
    """
    Aggregates analysis from multiple paragraphs into a final structured JSON object.
    Consolidates law firm information and determines if target company appears in any paragraph.
    The goal is to identify the representative law firms of involved parties and determine if the target company is mentioned, ensuring the results are structured and accurate.
    """
    paragraph_analyses = dspy.InputField(desc="List of extraction results from all paragraphs")
    
    buyer_firm = dspy.OutputField(desc="Law firm representing the Buyer (consolidated from all paragraphs)")
    seller_firm = dspy.OutputField(desc="Law firm representing the Seller (consolidated from all paragraphs)")
    third_party = dspy.OutputField(desc="Third-party law firm mentioned (consolidated from all paragraphs)")
    contains_target_firm = dspy.OutputField(desc="Boolean: true if target company is mentioned in any paragraph, false otherwise")


# ============================================================================
# DECISION: Prompt Type
# ============================================================================
# Reasoning:
# - LLM3's task is aggregation/consolidation, which requires understanding
#   which information to prioritize when multiple sources exist
# - Edge cases: multiple firms, conflicting info, "Not stated" handling
# - ChainOfThought helps model reason through consolidation logic
# - However, aggregation is more structured than extraction, so simpler might work
# - DECISION: Use ChainOfThought for better reasoning on edge cases
#   (We can test simpler Predict() later if needed)
# ============================================================================

# ============================================================================
# DECISION: Training Method
# ============================================================================
# Reasoning:
# - MIPROv2 is powerful but expensive and may be overkill for aggregation
# - BootstrapFewShot is simpler, faster, cheaper, and often sufficient
# - Aggregation has clear patterns that few-shot learning can capture
# - MIPROv2 would help optimize consolidation logic, but may not be necessary
# - DECISION: Start with BootstrapFewShot (faster/cheaper), with MIPROv2 as option
#   If BootstrapFewShot doesn't perform well, we can upgrade to MIPROv2
# ============================================================================

def format_paragraph_analyses(output_2_array):
    """
    Convert LLM2 output array to string format for LLM3.
    Format: Each paragraph analysis as a structured text block, separated by newlines.
    """
    formatted_parts = []
    for i, analysis in enumerate(output_2_array, 1):
        part = (
            f"Paragraph {i} Analysis:\n"
            f"Buyer: {analysis.get('Buyer', 'Not stated')}\n"
            f"Buyer Representative: {analysis.get('Buyer Representative', 'Not stated')}\n"
            f"Seller: {analysis.get('Seller', 'Not stated')}\n"
            f"Seller Representative: {analysis.get('Seller Representative', 'Not stated')}\n"
            f"Third-Party Representation: {analysis.get('Third-Party Representation', 'Not stated')}\n"
            f"Target Company Mentioned: {analysis.get('Target Company Mentioned', 'No')}"
        )
        formatted_parts.append(part)
    
    return "\n\n".join(formatted_parts)


# Create training examples
training_examples = []
for item in train_data:
    paragraph_analyses = format_paragraph_analyses(item['output_2'])
    answer = item['answer']
    training_examples.append(
        dspy.Example(
            paragraph_analyses=paragraph_analyses,
            buyer_firm=answer.get('buyer_firm', 'Not stated'),
            seller_firm=answer.get('seller_firm', 'Not stated'),
            third_party=answer.get('third_party', 'Not stated'),
            contains_target_firm=str(answer.get('contains_target_firm', False)).lower()
        ).with_inputs("paragraph_analyses")
    )

# Create validation examples (for MIPROv2) - must be dspy.Example objects
val_examples = []
val_size = min(10, len(test_data))  # Use up to 10 examples for validation
for item in test_data[:val_size]:
    paragraph_analyses = format_paragraph_analyses(item['output_2'])
    answer = item['answer']
    val_examples.append(
        dspy.Example(
            paragraph_analyses=paragraph_analyses,
            buyer_firm=answer.get('buyer_firm', 'Not stated'),
            seller_firm=answer.get('seller_firm', 'Not stated'),
            third_party=answer.get('third_party', 'Not stated'),
            contains_target_firm=str(answer.get('contains_target_firm', False)).lower()
        ).with_inputs("paragraph_analyses")
    )

# Create test examples (for final evaluation) - keep as dicts for easier testing
test_examples = []
for item in test_data:
    paragraph_analyses = format_paragraph_analyses(item['output_2'])
    answer = item['answer']
    test_examples.append({
        'paragraph_analyses': paragraph_analyses,
        'expected_output': {
            'buyer_firm': answer.get('buyer_firm', 'Not stated'),
            'seller_firm': answer.get('seller_firm', 'Not stated'),
            'third_party': answer.get('third_party', 'Not stated'),
            'contains_target_firm': str(answer.get('contains_target_firm', False)).lower()
        }
    })

# Create the module
class AggregationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning on consolidation logic
        self.aggregator = dspy.ChainOfThought(AggregateResults)
    
    def forward(self, paragraph_analyses):
        result = self.aggregator(paragraph_analyses=paragraph_analyses)
        return result

print("🔨 Training LLM3 with BootstrapFewShot optimizer...")
print("📚 Teacher (for optimization): GPT-4o")
print("📚 Student (production): GPT-4o-mini")
print("📝 Prompt Type: ChainOfThought (for better consolidation reasoning)\n")

# Create the module
print("🎓 Creating AggregationModule...")
module = AggregationModule()

# Validation metric
def validate_output(example, pred, trace=None):
    """
    Metric function: (example, pred, trace=None) -> float
    
    Returns a normalized score from 0.0 to 1.0 based on how many fields are correct.
    Each correct field = 1/4 points, normalized to 0.0-1.0 range.
    """
    fields = ['buyer_firm', 'seller_firm', 'third_party', 'contains_target_firm']
    
    max_score = len(fields)
    score = 0.0
    
    for field in fields:
        # Skip if field doesn't exist
        if not hasattr(pred, field):
            continue
        
        expected_val = str(getattr(example, field, "")).strip().lower()
        predicted_val = str(getattr(pred, field, "")).strip().lower()
        
        # Check if field is correct (case-insensitive matching)
        if expected_val == predicted_val:
            score += 1.0
    
    # Normalize to 0.0-1.0 range
    return score / max_score if max_score > 0 else 0.0

# Use BootstrapFewShot optimizer (simpler, faster, cheaper than MIPROv2)
# Can upgrade to MIPROv2 if needed for better performance
print("🏗️ Setting up BootstrapFewShot optimizer...")
print("   (Using BootstrapFewShot for faster/cheaper training)")
print("   (Can upgrade to MIPROv2 if performance needs improvement)\n")
compiled_model_path = "llm3_optimized.json"

if os.path.exists(compiled_model_path):
    print(f"📂 Loading optimized model from {compiled_model_path}...")
    module.load(compiled_model_path)
    optimized_program = module
    print("✅ Model loaded successfully!\n")
else:
    try:
        from dspy.teleprompt import BootstrapFewShot
        
        # BootstrapFewShot: simpler, faster, cheaper
        # It selects good few-shot examples and optimizes the prompt
        teleprompter = BootstrapFewShot(
            metric=validate_output,
            teacher=teacher_llm,  # Teacher generates examples
            max_bootstrapped_demos=4,  # Number of examples to include in prompt
            max_labeled_demos=16  # Max examples to consider
        )
        
        print("🔄 Compiling module with BootstrapFewShot...")
        print("   (This may take a few minutes and will use API calls)\n")
        
        optimized_program = teleprompter.compile(
            student=module,
            trainset=training_examples
        )
        
        print("✅ Module optimized successfully with BootstrapFewShot!\n")
        
    except ImportError:
        print("⚠️  BootstrapFewShot not available. Trying MIPROv2 as fallback...")
        try:
            from dspy.teleprompt import MIPROv2
            teleprompter = MIPROv2(
                metric=validate_output,
                prompt_model=teacher_llm,
                task_model=student_llm,
                auto="light",
                init_temperature=1.0
            )
            optimized_program = teleprompter.compile(
                student=module,
                trainset=training_examples,
                valset=val_examples
            )
            print("✅ Module optimized with MIPROv2!\n")
        except Exception as e:
            print(f"⚠️  Error with MIPROv2: {e}")
            print("   Falling back to basic module...")
            optimized_program = module
    except Exception as e:
        print(f"⚠️  Error with BootstrapFewShot: {e}")
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
    paragraph_analyses = test_item['paragraph_analyses']
    expected = test_item['expected_output']
    
    try:
        result = optimized_program(paragraph_analyses=paragraph_analyses)
        
        # Check if all fields match
        fields = ['buyer_firm', 'seller_firm', 'third_party', 'contains_target_firm']
        all_match = True
        
        for field in fields:
            expected_val = str(expected[field]).strip().lower()
            predicted_val = str(getattr(result, field, "")).strip().lower()
            
            # Flexible matching for "Not stated"
            if expected_val == "not stated":
                if predicted_val not in ["not stated", "none", "n/a", ""]:
                    all_match = False
            elif expected_val != predicted_val:
                all_match = False
        
        if all_match:
            correct += 1
        
        status = "✅" if all_match else "❌"
        print(f"\n{status} Test {i}/{total}")
        for field in fields:
            expected_val = expected[field]
            predicted_val = getattr(result, field, 'N/A')
            match_indicator = "✓" if str(expected_val).strip().lower() == str(predicted_val).strip().lower() else "✗"
            print(f"   {match_indicator} {field}: Expected '{expected_val}' | Got '{predicted_val}'")
        
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
