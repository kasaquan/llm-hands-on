import dspy
import json
import os
import random

# Load user queries (now includes expected outputs)
with open('user_queries.json', 'r') as f:
    user_queries_data = json.load(f)

# Shuffle the queries for random train/test split
random.shuffle(user_queries_data)

# Split into 70% training and 30% testing
total_queries = len(user_queries_data)
train_size = int(total_queries * 0.7)
train_data = user_queries_data[:train_size]
test_data = user_queries_data[train_size:]

print(f"📊 Dataset split:")
print(f"   Total queries: {total_queries}")
print(f"   Training (70%): {len(train_data)} queries")
print(f"   Testing (30%): {len(test_data)} queries\n")

# Configure DSPy with OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️  Warning: OPENAI_API_KEY not set. Please set it before running.")
    print("   export OPENAI_API_KEY='your-key-here'")

# Set up teacher (Architect) and student (Intern) models
# The "Intern" (Cheap, Fast)
student_llm = dspy.LM(model="openai/gpt-4o-mini", api_key=openai_api_key)

# The "Architect" (Smart, Expensive)
teacher_llm = dspy.LM(model="openai/gpt-4o", api_key=openai_api_key)

# Define the signature for relevance checking
class RelevanceCheck(dspy.Signature):
    """Check if a user query mentions a target company in a contract.
    If a target company is identified, output: 'The target company is [Company Name].'
    If no target company is found, output: '<user_message>Query is not relevant to the intended task.</user_message>'"""
    
    query = dspy.InputField(desc="The user's query about a contract")
    output = dspy.OutputField(desc="Either 'The target company is [Company Name].' or '<user_message>Query is not relevant to the intended task.</user_message>'")

# Create training examples from the training data (70%)
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

# Create test examples from the test data (30%)
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
        self.relevance_checker = dspy.ChainOfThought(RelevanceCheck)
    
    def forward(self, query):
        result = self.relevance_checker(query=query)
        return result.output

print("🔨 Using Architect & Intern pattern to generate LLM1 prompt...")
print("📚 Teacher (Architect): GPT-4o")
print("📚 Student (Intern): GPT-4o-mini\n")

# Create the student (Intern) module
print("🎓 Creating Intern (Student) module...")
intern_module = RelevanceModule()

# The Optimizer acts as the bridge
# This uses the Teacher to generate "Gold Standard" reasoning traces
# and feeds them into the Student's prompt structure.
print("🏗️ Setting up BootstrapFewShot optimizer with Architect as teacher...")
try:
    from dspy.teleprompt import BootstrapFewShot
    
    # Optional: Create a simple metric function if needed
    # For now, we'll use None and let the teacher generate examples
    def validate_output(example, pred, trace=None):
        """Simple validation metric"""
        return example.output.strip() == pred.output.strip() if hasattr(pred, 'output') else False
    
    # The Optimizer uses the Teacher to generate "Gold Standard" reasoning traces
    teleprompter = BootstrapFewShot(metric=validate_output, teacher=teacher_llm)
    
    print("🔄 Compiling Intern module with Architect's guidance...")
    print("   (This uses the Architect to generate gold-standard examples for the Intern)\n")
    
    # This uses the Teacher to generate "Gold Standard" reasoning traces
    # and feeds them into the Student's prompt structure.
    optimized_program = teleprompter.compile(student=intern_module, trainset=training_examples)
    
    print("✅ Intern module optimized successfully with Architect's guidance!\n")
    
except Exception as e:
    print(f"⚠️  BootstrapFewShot not available or error occurred: {e}")
    print("   Falling back to basic module...")
    optimized_program = intern_module

# Test the optimized intern module on the test set (30%)
print("🧪 Testing Optimized Intern (GPT-4o-mini) on test set:")
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