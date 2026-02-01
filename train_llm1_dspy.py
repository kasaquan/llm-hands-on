import dspy
import json
import os
import random
import warnings

# Suppress Pydantic serialization warnings (non-critical)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load user queries (now includes expected outputs)
with open('user_queries.json', 'r') as f:
    user_queries_data = json.load(f)

# Shuffle the queries for random train/validation/test split
random.shuffle(user_queries_data)

# Split into 60% training, 20% validation, 20% testing
total_queries = len(user_queries_data)
train_size = 60
val_size = 20
test_size = 20

train_data = user_queries_data[:train_size]
val_data = user_queries_data[train_size:train_size + val_size]
test_data = user_queries_data[train_size + val_size:train_size + val_size + test_size]

print(f"📊 Dataset split:")
print(f"   Total queries: {total_queries}")
print(f"   Training: {len(train_data)} examples (60%)")
print(f"   Validation: {len(val_data)} examples (20%)")
print(f"   Testing: {len(test_data)} examples (20%)\n")

# Configure DSPy with OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️  Warning: OPENAI_API_KEY not set. Please set it before running.")
    print("   export OPENAI_API_KEY='your-key-here'")

# Set up teacher (Architect) and student (Intern) models
# The "Intern" (Cheap, Fast)
student_llm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key, temperature=0.2)

# The "Architect" (Smart, Expensive)
teacher_llm = dspy.LM(model="openai/gpt-5.2", api_key=openai_api_key)

# Define the signature for relevance checking
class RelevanceCheck(dspy.Signature):
    """Determines if the user's query mentions any target company.
    1. If no target company is found, respond with a message wrapped in <user_message></user_message> XML tags to inform the user that the query is irrelevant to this task.
       Example: <user_message>Query is not relevant to the intended task.</user_message>
    2. If the query contains a target company, respond with a formatted acknowledgment of the identified target company: 'The target company is [Company Name].'
    """
    
    query = dspy.InputField(desc="The user's query about a contract")
    output = dspy.OutputField(desc="The final answer. Use predefined <user_message> for negative cases.")

# Create training examples from the training data (60%)
training_examples = []
for item in train_data:
    if isinstance(item, dict):
        training_examples.append(
            dspy.Example(query=item['query'], output=item['expected_output']).with_inputs("query")
        )
    else:
        # Fallback for old format
        training_examples.append(
            dspy.Example(query=item, output="<user_message>Query is not relevant to the intended task.</user_message>").with_inputs("query")
        )

# Create validation examples from the validation data (20%)
validation_examples = []
for item in val_data:
    if isinstance(item, dict):
        validation_examples.append(
            dspy.Example(query=item['query'], output=item['expected_output']).with_inputs("query")
        )
    else:
        # Fallback for old format
        validation_examples.append(
            dspy.Example(query=item, output="<user_message>Query is not relevant to the intended task.</user_message>").with_inputs("query")
        )

# Create test examples from the test data (20%)
test_examples = []
for item in test_data:
    if isinstance(item, dict):
        test_examples.append({
            'query': item['query'],
            'expected_output': item['expected_output']
        })
    else:
        test_examples.append({
            'query': item,
            'expected_output': "<user_message>Query is not relevant to the intended task.</user_message>"
        })

# Create the module
class RelevanceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use dspy.Predict instead of ChainOfThought to avoid reasoning output
        # The online eval system expects direct output without reasoning
        self.relevance_checker = dspy.Predict(RelevanceCheck)
    
    def forward(self, query):
        result = self.relevance_checker(query=query)            
        return result.output

print("🔨 Using Architect & Intern pattern to generate LLM1 prompt...")
print("📚 Teacher (Architect): GPT-4o")
print("📚 Student (Intern): GPT-4o-mini\n")

# Create the student (Intern) module
print("🎓 Creating Intern (Student) module...")
intern_module = RelevanceModule()

# Validation metric (MIPROv2 standard format)
def validate_output(example, pred, trace=None):
    """
    MIPROv2 metric function signature: (example, pred, trace=None) -> bool/float
    
    - example: dspy.Example with expected output in example.output
    - pred: dspy.Prediction object with pred.output (matches RelevanceCheck signature)
    - trace: optional trace object (unused here)
    
    Returns: True/False or 1.0/0.0
    """
    # MIPROv2 passes the prediction object from the Signature
    # Since RelevanceCheck has 'output' field, pred should have pred.output
    # However, RelevanceModule.forward returns the string output directly, so we need to handle both cases
    if hasattr(pred, 'output'):
        predicted = str(pred.output).strip() if pred.output is not None else ""
    else:
        predicted = str(pred).strip()
    
    expected = str(example.output).strip()
    
    # 1. Exact match
    if predicted == expected:
        return True
        
    # 2. Handle XML tag hallucinations (e.g. missing tags)
    # If expected is the negative case, check if the core message is present
    if "not relevant" in expected.lower() and "not relevant" in predicted.lower():
        return True
        
    # 3. Handle company name extraction (e.g. "The target company is [X]")
    # Extract just the company name and compare
    if "The target company is" in expected:
        company = expected.replace("The target company is", "").strip(" .")
        if company and company.lower() in predicted.lower():
            return True
        
    return False

# Use MIPROv2 optimizer for production with gpt-4o-mini
print("🏗️ Setting up MIPROv2 optimizer...")
compiled_model_path = "llm1_optimized.json"

if os.path.exists(compiled_model_path):
    print(f"📂 Loading optimized model from {compiled_model_path}...")
    intern_module.load(compiled_model_path)
    optimized_program = intern_module
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
            student=intern_module,
            trainset=training_examples,
            valset=validation_examples  # Separate validation set (20 examples)
        )
        
        print("✅ Module optimized successfully with MIPROv2!\n")
        
    except ImportError:
        print("⚠️  MIPROv2 not available. Trying BootstrapFewShot as fallback...")
        from dspy.teleprompt import BootstrapFewShot
        teleprompter = BootstrapFewShot(metric=validate_output, teacher=teacher_llm)
        optimized_program = teleprompter.compile(student=intern_module, trainset=training_examples)
        print("✅ Module optimized with BootstrapFewShot!\n")
    except Exception as e:
        print(f"⚠️  Error with MIPROv2: {e}")
        print("   Falling back to basic module...")
        optimized_program = intern_module

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
    query = test_item['query']
    expected = test_item['expected_output']
    try:
        result = optimized_program(query)
        is_correct = result.strip() == expected.strip()
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Test {i}/{total}: {query}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
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