import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Dungeon Story Generator - Pro Edition")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model Setup ---
# We use a small instruction-tuned model for better control over the story and choices.
# Qwen2.5-0.5B-Instruct is very capable of following structures and generating good prose.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

logger.info(f"Loading model: {MODEL_NAME}")
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Ensure a pad token exists. Some tokenizers don't define one and set pad==eos,
    # which leads to an attention-mask warning. Set pad_token to eos if missing.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer had no pad_token; set pad_token = eos_token")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.to(device)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.warning(f"Failed to load full model, trying pipeline fallback: {e}")
    try:
        generator = pipeline("text-generation", model="gpt2")
        model = None
    except Exception as e2:
        logger.error(f"Failed to load any model: {e2}")
        model = None


class StoryContext(BaseModel):
    genre: str
    character_name: str
    character_details: str
    lore: str

class StartStoryRequest(BaseModel):
    context: StoryContext
    starter_prompt: str

class ContinueStoryRequest(BaseModel):
    context: StoryContext
    history: str
    user_choice: str

class StoryResponse(BaseModel):
    story_chunk: str
    choices: List[str]


def parse_output(text: str):
    """
    Parses the model output to extract the narrative and choices.
    Expects format:
    [Story]
    ...
    [Choices]
    1. ...
    2. ...
    3. ...
    """
    story_chunk = ""
    choices = []
    
    # Try to parse with explicit markers
    if "[Story]" in text and "[Choices]" in text:
        parts = text.split("[Choices]")
        story_chunk = parts[0].replace("[Story]", "").strip()
        choices_text = parts[1].strip()
        
        # Extract numbered choices
        lines = choices_text.split("\n")
        for line in lines:
            line = line.strip()
            if re.match(r"^\d+\.", line) or line.startswith("-"):
                choice = re.sub(r"^(\d+\.|-)\s*", "", line).strip()
                if choice:
                    choices.append(choice)
    else:
        # Fallback if no clear markers
        story_chunk = text.strip()
        choices = ["Explore the area carefully", "Act recklessly", "Wait and observe"]
        
    # Ensure we always have 3 choices
    while len(choices) < 3:
        choices.append("Do something unpredictable")
        
    return story_chunk, choices[:3]


def generate_pro_story(prompt: str) -> str:
    if model is None:
        return "[Story] You walk forward into the unknown.\n[Choices]\n1. Keep going\n2. Turn back\n3. Rest"
        
    messages = [
        {"role": "system", "content": "You are a master storyteller, a 'Pro Writer'. Your writing is highly descriptive, visceral, and uses 'Show, Don't Tell'. Write in second person ('You'). Provide the next part of the story, followed exactly by three distinct choices for the player."},
        {"role": "user", "content": prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and ensure attention mask is returned; move tensors to the device.
    model_inputs = tokenizer([text_input], return_tensors="pt", padding=True, truncation=True)
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    # Generate the response
    generated_ids = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs.get("attention_mask", None),
        max_new_tokens=300,
        temperature=0.85, # slightly creative
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


@app.post("/api/start", response_model=StoryResponse)
async def start_story(req: StartStoryRequest):
    prompt = f"""
Genre: {req.context.genre}
Character: {req.context.character_name}. {req.context.character_details}
Lore: {req.context.lore}

Action: The adventure begins. {req.starter_prompt}

Write the opening narrative (around 3 paragraphs) describing the scene, establishing the atmosphere, and putting the character in the moment. Then, provide exactly three choices for what the character should do next.
Format strictly as:
[Story]
<your story here>

[Choices]
1. <choice 1>
2. <choice 2>
3. <choice 3>
"""
    try:
        raw_output = generate_pro_story(prompt)
        story_chunk, choices = parse_output(raw_output)
        return StoryResponse(story_chunk=story_chunk, choices=choices)
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate story.")


@app.post("/api/continue", response_model=StoryResponse)
async def continue_story(req: ContinueStoryRequest):
    prompt = f"""
Genre: {req.context.genre}
Character: {req.context.character_name}. {req.context.character_details}
Lore: {req.context.lore}

Previous Story Context:
{req.history[-1000:]}

User Choice: {req.user_choice}

Write the next part of the story (2-3 paragraphs) based on the user's choice. Continue to use vivid, sensory language. Build tension or progress the plot. Then, provide exactly three new choices for what the character should do next.
Format strictly as:
[Story]
<your story here>

[Choices]
1. <choice 1>
2. <choice 2>
3. <choice 3>
"""
    try:
        raw_output = generate_pro_story(prompt)
        story_chunk, choices = parse_output(raw_output)
        return StoryResponse(story_chunk=story_chunk, choices=choices)
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate story.")

# Serve the static frontend
import os
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
