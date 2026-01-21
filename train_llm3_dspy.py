import dspy
import json
import os
import random
import warnings
from dspy.teleprompt import BootstrapFewShot

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

# Configure default LM to student
dspy.configure(lm=teacher_llm)

class AggregateResults(dspy.Signature):
    """
    Aggregates analysis from multiple paragraphs into a final structured JSON object.
    Consolidates law firm information and determines if target company appears in any paragraph.
    The goal is to identify the representative law firms of involved parties and determine if the target company is mentioned, ensuring the results are structured and accurate.
    """
    paragraph_analyses = dspy.InputField(desc="Concatenated strings of several paragraph analyses.")
    json_output = dspy.OutputField(desc="A valid JSON string with exactly 4 fields: buyer_firm (string), seller_firm (string), third_party (string), contains_target_firm (boolean). Output ONLY the JSON string, no other text.")

# Create the module
class AggregationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning on consolidation logic
        self.aggregator = dspy.ChainOfThought(AggregateResults)
    
    def forward(self, paragraph_analyses):
        # LLM outputs JSON directly - no post-processing needed
        return self.aggregator(paragraph_analyses=paragraph_analyses)

# Helper functions for data processing
def format_inputs(inputs):
    """Concatenate items from 'inputs' as a string."""
    return "\n".join([json.dumps(item) for item in inputs])

def create_example(item):
    """Create a dspy.Example from a data item."""
    paragraph_analyses = format_inputs(item['inputs'])
    # Use 'answer' field as expected response (converted to JSON string)
    json_output = json.dumps(item['answer'])
    
    return dspy.Example(
        paragraph_analyses=paragraph_analyses,
        json_output=json_output
    ).with_inputs("paragraph_analyses")

# Generate training and evaluation sets
print("🔄 Generating training and evaluation examples...")
training_examples = [create_example(item) for item in train_data]
eval_set = [create_example(item) for item in test_data]
print(f"   Training examples: {len(training_examples)}")
print(f"   Evaluation examples: {len(eval_set)}\n")

print("🔨 Training LLM3 with BootstrapFewShot optimizer...")
print("📚 Teacher (for optimization): GPT-4o")
print("📚 Student (production): GPT-4o-mini")
print("📝 Prompt Type: ChainOfThought (for better consolidation reasoning)\n")

# Create the module
print("🎓 Creating AggregationModule...")
module = AggregationModule()

# Validation metric
def validate_output(example, pred, trace=None):
    required_keys = ['buyer_firm', 'seller_firm', 'third_party', 'contains_target_firm']
    
    # Parse expected JSON from example
    try:
        expected_json = json.loads(getattr(example, 'json_output', '{}'))
    except json.JSONDecodeError:
        return 0.0
    
    parsed_result = json.loads(pred.json_output)
    # Check if all required keys are present
    missing_keys = [key for key in required_keys if key not in parsed_result]
    if missing_keys:
        return 0.0
    
    # Compare values for each required key (exact matching)
    incorrect_values = []
    for key in required_keys:
        if parsed_result[key] != expected_json[key]:
            incorrect_values.append(key)
    
    return (len(required_keys) - len(incorrect_values)) / len(required_keys)

# Use BootstrapFewShot optimizer (simpler, faster, cheaper than MIPROv2)
# Can upgrade to MIPROv2 if needed for better performance
print("🏗️ Setting up BootstrapFewShot optimizer...")
compiled_model_path = "llm3_optimized.json"

if os.path.exists(compiled_model_path):
    print(f"📂 Loading optimized model from {compiled_model_path}...")
    module.load(compiled_model_path)
    optimized_program = module
    print("✅ Model loaded successfully!\n")
else:
    try:
        
        teleprompter = BootstrapFewShot(
            metric=validate_output
        )
        print("🔄 Compiling module with BootstrapFewShot...")
        print("   (This may take a few minutes and will use API calls)\n")
        
        optimized_program = teleprompter.compile(
            student=module,
            trainset=training_examples,
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
                valset=eval_set[:10]
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

# Test the optimized module on the eval set
print("🧪 Testing Optimized Module (GPT-4o-mini) on eval set:")
print("=" * 60)
dspy.configure(lm=student_llm)  # Use student LM for inference

correct = 0
total = len(eval_set)

for i, example in enumerate(eval_set, 1):
    try:
        pred = optimized_program(paragraph_analyses=example.paragraph_analyses)
        score = validate_output(example, pred)
        
        if score == 1.0:
            correct += 1
            status = "✅"
        else:
            status = "❌"
            
        print(f"{status} Test {i}/{total} (Score: {score:.2f})")
        
        # Detailed comparison for failures
        if score < 1.0:
            try:
                expected_json = json.loads(example.json_output)
                
                # Extract JSON from prediction more robustly
                pred_json_str = getattr(pred, 'json_output', '{}')
                # Try to find JSON block if mixed with text
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', pred_json_str, re.DOTALL)
                if json_match:
                    pred_json_str = json_match.group(0)
                
                try:
                    pred_json = json.loads(pred_json_str)
                except json.JSONDecodeError:
                    pred_json = {"error": "Could not parse JSON", "raw": pred_json_str}

                print("   ⚠️  Mismatch details:")
                required_keys = ['buyer_firm', 'seller_firm', 'third_party', 'contains_target_firm']
                for key in required_keys:
                    exp_val = expected_json.get(key, "MISSING")
                    pred_val = pred_json.get(key, "MISSING")
                    
                    if exp_val != pred_val:
                        print(f"      ❌ {key}: Expected '{exp_val}' | Got '{pred_val}'")
                    else:
                        print(f"      ✅ {key}: '{exp_val}'")
                        
            except Exception as parse_err:
                print(f"   ⚠️  Error parsing for comparison: {parse_err}")
                print(f"   Expected: {example.json_output}")
                print(f"   Predicted: {getattr(pred, 'json_output', 'No output')}")

    except Exception as e:
        print(f"❌ Error with test {i}: {e}")

print("\n" + "=" * 60)
print(f"📊 Test Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("=" * 60)

# Inspect the last prompt sent to the student model
print("\n🔍 Inspecting the last prompt sent to the Student LLM:")
print("=" * 60)
student_llm.inspect_history(n=1)
print("=" * 60)
