from openai import OpenAI
client = OpenAI()

# response = client.responses.create(
#     model="gpt-5-nano",
#     input="Write a one-sentence bedtime story about a unicorn."
# )

# print(response.output_text)

MODEL_NAME = "gpt-4o-mini"  # Ensure this matches your deployed model.
TEMPERATURE = 0.2

#load system prompt 1
with open('llm1_prompt.txt', 'r') as file:
    system_prompt_1 = file.read()
# print(system_prompt_1)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,  # Ensure this model identifier matches your deployed model.
        messages=[
            {"role": "system", "content": system_prompt_1},
            {"role": "user", "content": "Is Kirkland & Ellis present in the agreement?"}
        ],
        temperature=TEMPERATURE
    )
    output1 = response.choices[0].message.content.strip()
    print(output1)
except Exception as e:
    output1 = f"Error during OpenAI API call: {str(e)}"
