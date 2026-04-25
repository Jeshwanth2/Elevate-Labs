from transformers import pipeline

# Load text generation pipeline with GPT-2
generator = pipeline("text-generation", model="gpt2")


genre = "fantasy"  # try mystery, horror later
user_input = "A young warrior finds a glowing sword in the forest."

prompt = f"In this {genre} tale, {user_input}"

# Generate story continuation
output = generator(
    prompt,
    max_new_tokens=120,
    num_return_sequences=3,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)
for i, story in enumerate(output):
    print(f"\n--- Story {i+1} ---\n")
    print(story["generated_text"])