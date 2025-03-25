# %%
###########################################
# Complete End-to-End NutrionBot with Disambiguation and Improved Responses
###########################################

import os
import re
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# -------------------------------
# 1. Set Up OpenAI API Key Securely
# -------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please set it with a valid key.")

# -------------------------------
# 2. Load and Clean the Nutrition Dataset
# -------------------------------
dataset_folder = "Dataset"
csv_file_path = os.path.join(dataset_folder, "FOOD-DATA-MERGED_CLEANED.csv")

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder, exist_ok=True)

try:
    data = pd.read_csv(csv_file_path)
except Exception as e:
    raise FileNotFoundError(f"Error loading {csv_file_path}: {e}")

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

if 'food' in data.columns:
    data['food'] = data['food'].str.lower().str.strip()
else:
    print("Warning: 'food' column not found in dataset!")

data = data.drop_duplicates(subset=['food'])
numeric_cols = data.columns.drop('food')
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data[numeric_cols] = data[numeric_cols].fillna(0)

food_dict = {row['food']: row.drop('food').to_dict() for _, row in data.iterrows()}

# -------------------------------
# 3. Nutrient Synonym Dictionary
# -------------------------------
nutrient_synonyms = {
    "calories": "Caloric Value",
    "caloric value": "Caloric Value",
    "fat": "Fat",
    "saturated fat": "Saturated Fats",
    "saturated fats": "Saturated Fats",
    "monounsaturated fat": "Monounsaturated Fats",
    "monounsaturated fats": "Monounsaturated Fats",
    "polyunsaturated fat": "Polyunsaturated Fats",
    "polyunsaturated fats": "Polyunsaturated Fats",
    "carbs": "Carbohydrates",
    "carbohydrate": "Carbohydrates",
    "carbohydrates": "Carbohydrates",
    "sugar": "Sugars",
    "sugars": "Sugars",
    "protein": "Protein",
    "dietary fiber": "Dietary Fiber",
    "fiber": "Dietary Fiber",
    "cholesterol": "Cholesterol",
    "sodium": "Sodium",
    "water": "Water",
    "vitamin a": "Vitamin A",
    "vitamin b1": "Vitamin B1",
    "thiamine": "Vitamin B1",
    "vitamin b11": "Vitamin B11",
    "vitamin b12": "Vitamin B12",
    "vitamin b2": "Vitamin B2",
    "riboflavin": "Vitamin B2",
    "vitamin b3": "Vitamin B3",
    "niacin": "Vitamin B3",
    "vitamin b5": "Vitamin B5",
    "vitamin b6": "Vitamin B6",
    "vitamin c": "Vitamin C",
    "vitamin d": "Vitamin D",
    "vitamin e": "Vitamin E",
    "vitamin k": "Vitamin K",
    "calcium": "Calcium",
    "copper": "Copper",
    "iron": "Iron",
    "magnesium": "Magnesium",
    "manganese": "Manganese",
    "phosphorus": "Phosphorus",
    "potassium": "Potassium",
    "selenium": "Selenium",
    "zinc": "Zinc",
    "nutrition density": "Nutrition Density",
    "nutritional density": "Nutrition Density"
}

def find_nutrient_in_query(query_lower):
    matched = []
    for user_term, column_name in nutrient_synonyms.items():
        if user_term in query_lower:
            matched.append(column_name)
    return list(set(matched))

# -------------------------------
# 4. TF-IDF Retrieval and Helpers
# -------------------------------
def dict_to_markdown_table(info):
    """Convert a dictionary of nutritional info into a Markdown table."""
    md = "| Nutrient | Value |\n|----------|-------|\n"
    for k, v in info.items():
        md += f"| {k} | {v} |\n"
    return md

def retrieve_results(query, top_n=5):
    keys = list(food_dict.keys())
    documents = []
    for f in keys:
        info_str = ", ".join([f"{k}: {v}" for k, v in food_dict[f].items()])
        documents.append(f"{f}. {info_str}")
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "FoodName": keys[idx],
            "Nutrition": food_dict[keys[idx]],
            "similarity": float(sims[idx])
        })
    return results

# -------------------------------
# 5. Multi-step Disambiguation Logic
# -------------------------------
def match_user_choice_to_candidates(user_input, candidates):
    """
    Match user input to a candidate list (by number or exact name).
    """
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
        else:
            return None
    user_input_lower = user_input.lower().strip()
    for c in candidates:
        if user_input_lower == c.lower():
            return c
    return None

def handle_final_disambiguation(food_item, matched_nutrients, original_query):
    info = food_dict.get(food_item)
    if not info:
        return f"Sorry, no nutritional info for {food_item}."
    if matched_nutrients:
        nutrient_value = None
        for k in info.keys():
            for nutrient in matched_nutrients:
                if nutrient.lower() == k.lower():
                    nutrient_value = info[k]
                    break
            if nutrient_value is not None:
                break
        if nutrient_value is not None:
            # Even if value is 0, display it with a friendly note and provide the full table.
            if nutrient_value == 0:
                answer = f"The {matched_nutrients[0]} of {food_item} is 0 per serving. " \
                         f"Here is the full nutritional info for {food_item}:\n" + dict_to_markdown_table(info)
            else:
                answer = f"The {matched_nutrients[0]} of {food_item} is {nutrient_value} per serving."
            return answer
        else:
            answer = f"I couldn't specifically determine the {', '.join(matched_nutrients)} for {food_item}. " \
                     f"Here is the full nutritional info for {food_item}:\n" + dict_to_markdown_table(info)
            return answer
    else:
        return dict_to_markdown_table(info)

def generate_answer_extended(query):
    """
    Processes the query and returns a triple:
      (food_item or None, final_answer, disambig_info or None)
    If multiple items are found, disambiguation info is returned.
    """
    query_lower = query.lower().strip()
    matched_nutrients = find_nutrient_in_query(query_lower)
    
    # EXACT MATCH CHECK
    if query_lower in food_dict:
        food_item = query_lower
        info = food_dict[food_item]
        if matched_nutrients:
            nutrient_value = None
            for k in info.keys():
                for nutrient in matched_nutrients:
                    if nutrient.lower() == k.lower():
                        nutrient_value = info[k]
                        break
                if nutrient_value is not None:
                    break
            if nutrient_value is not None:
                if nutrient_value == 0:
                    answer = f"The {matched_nutrients[0]} of {food_item} is 0 per serving. " \
                             f"Here is the full nutritional info for {food_item}:\n" + dict_to_markdown_table(info)
                else:
                    answer = f"The {matched_nutrients[0]} of {food_item} is {nutrient_value} per serving."
                return (food_item, answer, None)
            else:
                answer = f"I couldn't specifically determine the {', '.join(matched_nutrients)} for {food_item}. " \
                         f"Here is the full nutritional info for {food_item}:\n" + dict_to_markdown_table(info)
                return (food_item, answer, None)
        else:
            answer = dict_to_markdown_table(info)
            return (food_item, answer, None)
    
    # Otherwise, use TF-IDF retrieval.
    retrieved = retrieve_results(query_lower, top_n=5)
    if not retrieved:
        return (None, "I couldn't find any matching foods. Please try a different query.", None)
    
    retrieved = sorted(retrieved, key=lambda x: x["similarity"], reverse=True)
    top1 = retrieved[0]
    sim1 = top1["similarity"]
    
    close_threshold = 0.1
    possible_candidates = [top1["FoodName"]]
    for r in retrieved[1:]:
        if abs(r["similarity"] - sim1) < close_threshold:
            possible_candidates.append(r["FoodName"])
    
    if len(possible_candidates) > 1:
        return (None, None, {
            "candidates": possible_candidates,
            "nutrients": matched_nutrients
        })
    
    food_item = top1["FoodName"]
    info = food_dict.get(food_item)
    if not info:
        return (None, f"Found '{food_item}' but no nutritional info is available.", None)
    if sim1 < 0.2:
        return (None, "I'm not sure which food you mean. Could you be more specific?", None)
    
    if matched_nutrients:
        nutrient_value = None
        for k in info.keys():
            for nutrient in matched_nutrients:
                if nutrient.lower() == k.lower():
                    nutrient_value = info[k]
                    break
            if nutrient_value is not None:
                break
        if nutrient_value is not None:
            if nutrient_value == 0:
                answer = f"The {matched_nutrients[0]} of {food_item} is 0 per serving. " \
                         f"Here is the full nutritional info for {food_item}:\n" + dict_to_markdown_table(info)
            else:
                answer = f"The {matched_nutrients[0]} of {food_item} is {nutrient_value} per serving."
            return (food_item, answer, None)
        else:
            answer = f"I couldn't specifically determine the {', '.join(matched_nutrients)} for {food_item}. " \
                     f"Here is the full nutritional info for {food_item}:\n" + dict_to_markdown_table(info)
            return (food_item, answer, None)
    else:
        answer = dict_to_markdown_table(info)
        return (food_item, answer, None)

# -------------------------------
# 6. Image Classification Placeholder
# -------------------------------
def classify_image(image):
    """
    A placeholder function for image classification.
    Currently, it simply prints the image size.
    Future integration can involve models (e.g., Hugging Face's ViT) to classify food.
    """
    try:
        from transformers import AutoProcessor, AutoModelForImageClassification
        from PIL import Image
        processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
        model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        return predicted_class
    except Exception as e:
        print("Error in image classification:", e)
        return None

# -------------------------------
# 7. Define Chatbot Interface Function
# -------------------------------
def chatbot_interface(user_input, user_image, state):
    """
    Processes the user's text query (and optional image) and returns the chatbot's response.
    Handles multi-step disambiguation using session state.
    """
    if user_image is not None:
        print("User uploaded an image of size:", user_image.size)
        # Future integration: process the image to extract food details.
    
    if state and "pending_disambiguation" in state and state["pending_disambiguation"] is not None:
        disambig = state["pending_disambiguation"]
        chosen_food = match_user_choice_to_candidates(user_input, disambig["candidates"])
        if chosen_food is None:
            return ("Please pick one from:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(disambig["candidates"])), state)
        final_answer = handle_final_disambiguation(chosen_food, disambig["nutrients"], disambig["original_query"])
        state["pending_disambiguation"] = None
        return final_answer, state

    food_item, answer, disambig_info = generate_answer_extended(user_input)
    if disambig_info is not None:
        state["pending_disambiguation"] = {
            "candidates": disambig_info["candidates"],
            "nutrients": disambig_info["nutrients"],
            "original_query": user_input
        }
        msg = (
            "I found multiple items for your query. Please pick one:\n"
            + "\n".join(f"{i+1}. {c}" for i, c in enumerate(disambig_info["candidates"]))
        )
        return msg, state
    else:
        return answer, state

# -------------------------------
# 8. Gradio UI Setup with FAQ and Feedback Tabs
# -------------------------------
def submit_feedback(feedback):
    print("User feedback:", feedback)
    return "Thank you for your feedback!"

chat_tab_greeting = """
**Hello and welcome to NutrionBot!**  
I'm your friendly guide to discovering nutritional facts about individual food items.  
Simply type your question below and I'll do my best to help.  
Have fun exploring NutrionBot!
"""

faq_text = """
## Frequently Asked Questions

**Q: What kind of food items can I query?**  
A: NutrionBot is designed for individual food items from our dataset (e.g., apple, cream cheese low fat, tofu). It may not work as well for composite dishes.

**Q: What nutritional information does NutrionBot provide?**  
A: It retrieves details for 34 nutritional parameters, such as Caloric Value, Fat, Protein, Dietary Fibre, Cholesterol, Sodium, Vitamins, Minerals, and Nutrition Density.

**Q: How should I phrase my question?**  
A: You can ask questions in plain English! For best results, include both the nutrient and the food item. For example:  
- "What are the calories in an apple?"  
- "How much protein does tofu have?"  
- "What is the nutritional density of cream cheese?"

**Q: Any tips for better results?**  
A: Since NutrionBot is designed for single food items, please type the common name of the food (e.g., "apple", "cream cheese low fat", or "tofu").
"""

with gr.Blocks() as demo:
    gr.Markdown("# NutrionBot with Disambiguation")
    state = gr.State({"pending_disambiguation": None})

    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.Markdown(chat_tab_greeting)
            chat_input = gr.Textbox(lines=2, label="Your Question")
            chat_image = gr.Image(label="Upload an image (optional)", type="pil")
            chat_output = gr.Markdown(label="Response")
            chat_button = gr.Button("Submit", variant="primary")
            chat_button.click(
                fn=chatbot_interface,
                inputs=[chat_input, chat_image, state],
                outputs=[chat_output, state]
            )
        with gr.Tab("FAQ"):
            gr.Markdown(faq_text)
        with gr.Tab("Feedback"):
            feedback_input = gr.Textbox(lines=4, label="Your Feedback")
            feedback_output = gr.Textbox(label="Feedback Response")
            feedback_button = gr.Button("Submit Feedback", variant="primary")
            feedback_button.click(fn=submit_feedback, inputs=[feedback_input], outputs=[feedback_output])

demo.launch(debug=True, share=True)



