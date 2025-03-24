# %%
###########################################
# Complete End-to-End NutrionBot
###########################################

# -------------------------------
# 1. Import Required Modules
# -------------------------------
import os
import re
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# -------------------------------
# 2. Set Up OpenAI API Key Securely
# -------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please set it with a valid key.")

# -------------------------------
# 3. Load and Clean the Nutrition Dataset
# -------------------------------
dataset_folder = "Dataset"
csv_file_path = os.path.join(dataset_folder, "FOOD-DATA-MERGED_CLEANED.csv")

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder, exist_ok=True)

try:
    data = pd.read_csv(csv_file_path)
except Exception as e:
    raise FileNotFoundError(f"Error loading {csv_file_path}: {e}")

# Remove any 'Unnamed' columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Standardise the 'food' column: lowercase and trim spaces
if 'food' in data.columns:
    data['food'] = data['food'].str.lower().str.strip()
else:
    print("Warning: 'food' column not found in dataset!")

# Remove duplicate food entries
data = data.drop_duplicates(subset=['food'])

# Convert numeric columns (except 'food') to numbers and fill missing values with 0
numeric_cols = data.columns.drop('food')
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data[numeric_cols] = data[numeric_cols].fillna(0)

print("Merged dataset shape:", data.shape)
example_food = data['food'].iloc[0]
print(f"Example food entry for '{example_food}':")
print(data[data['food'] == example_food].to_dict(orient="records"))

# Build a lookup dictionary: key = food name, value = nutritional info as a dictionary
food_dict = {row['food']: row.drop('food').to_dict() for _, row in data.iterrows()}

# Optionally, save the cleaned dataset back to file
output_path = os.path.join(dataset_folder, "FOOD-DATA-MERGED_CLEANED.csv")
data.to_csv(output_path, index=False)
print(f"Cleaned dataset saved as '{output_path}'.")

# -------------------------------
# 4. Nutrient Synonym Dictionary
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
    """Return a list of nutrient column names found in the query."""
    matched = []
    for user_term, column_name in nutrient_synonyms.items():
        if user_term in query_lower:
            matched.append(column_name)
    return list(set(matched))

# -------------------------------
# 5. Helper Functions for Answer Generation
# -------------------------------
def dict_to_markdown_table(info):
    """Convert a dictionary of nutritional info into a Markdown table."""
    md = "| Nutrient | Value |\n|----------|-------|\n"
    for k, v in info.items():
        md += f"| {k} | {v} |\n"
    return md

def retrieve_results(query, top_n=3):
    """Uses TF-IDF and cosine similarity to retrieve the best matching food entries."""
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

def generate_conversational_answer(query, context):
    """Calls GPT-4-0314 to generate a friendly answer."""
    try:
        prompt = (
            f"Here is the nutritional data:\n{context}\n"
            f"Please provide a detailed, friendly answer to the following question:\n{query}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4-0314",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message['content']
    except Exception as e:
        print("OpenAI API error:", e)
        return None

def normal_rag_answer(query, food_name, info):
    """Fallback method: generates a general answer using GPT-4."""
    context_str = ", ".join([f"{k}: {v}" for k, v in info.items()])
    prompt = (f"Here is the nutritional data for {food_name}: {context_str}.\n"
              f"Answer the following question: {query}")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        final_answer = response.choices[0].message.content.strip()
        return (food_name, final_answer)
    except Exception as e:
        return (food_name, f"Error calling GPT: {e}")

def generate_answer(query):
    """
    Processes the user's query to detect nutrient requests and match a food.
    Returns a final answer.
    """
    query_lower = query.lower().strip()
    matched_nutrients = find_nutrient_in_query(query_lower)
    
    # Attempt substring matching for the food name.
    partial_matches = [food for food in food_dict.keys() if food in query_lower or query_lower in food]
    
    if len(partial_matches) == 1:
        food_item = partial_matches[0]
        print(f"[DEBUG] Substring-based match found: {food_item}")
    elif len(partial_matches) > 1:
        matches_str = ", ".join(partial_matches[:10])
        return (None, f"I found multiple foods: {matches_str}. Please specify which one you mean.")
    else:
        retrieved = retrieve_results(query, top_n=3)
        if not retrieved:
            return (None, "I couldn't find any matching foods. Please try a different query.")
        retrieved = sorted(retrieved, key=lambda x: x["similarity"], reverse=True)
        top1 = retrieved[0]
        sim1 = top1["similarity"]
        sim2 = retrieved[1]["similarity"] if len(retrieved) > 1 else 0.0
        confidence_threshold = 0.2
        separation_threshold = 0.05
        if sim1 < confidence_threshold:
            return (None, "I'm not sure which food you mean. Could you be more specific?")
        if (sim1 - sim2) < separation_threshold and len(retrieved) > 1:
            candidates = [item["FoodName"] for item in retrieved]
            return (None, f"I found multiple possible matches: {', '.join(candidates)}. Please specify which one you meant.")
        food_item = top1["FoodName"]
    
    info = food_dict.get(food_item)
    if not info:
        return (None, f"Found '{food_item}' but no nutritional info is available, sorry.")
    
    # If the query contains a nutrient keyword, provide that specific info.
    if matched_nutrients:
        nutrient_value = None
        for k in info.keys():
            if any(nutrient in k.lower() for nutrient in matched_nutrients):
                nutrient_value = info[k]
                break
        if nutrient_value is not None:
            basic_answer = f"The {matched_nutrients[0]} of {food_item} is {nutrient_value} per serving."
        else:
            basic_answer = f"Sorry, I could not find information on {', '.join(matched_nutrients)} for {food_item}."
    else:
        # If no specific nutrient is requested, display all nutritional info in a table format.
        basic_answer = dict_to_markdown_table(info)
        # Return immediately since we want a table output.
        return (food_item, basic_answer)
    
    # Optionally, you could call GPT for a more conversational answer if desired.
    retrieved_context = retrieve_results(food_item, top_n=3)
    context_str = "\n".join([
        f"{item['FoodName']}: " + ", ".join([f"{k}: {v}" for k, v in item['Nutrition'].items()]) +
        f" (similarity: {item['similarity']:.2f})"
        for item in retrieved_context
    ])
    
    final_answer = generate_conversational_answer(query, context_str)
    if not final_answer or "error" in final_answer.lower():
        final_answer = basic_answer
    
    return (food_item, final_answer)

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
def chatbot_interface(user_input, user_image):
    """
    Processes the user's text query (and optional image) and returns the chatbot's response.
    """
    try:
        if user_image is not None:
            print("User uploaded an image of size:", user_image.size)
            # Future integration: process the image to extract food details.
        _, answer = generate_answer(user_input)
        return answer
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error occurred: {e}"

# -------------------------------
# 8. Gradio Interface Setup with FAQ and Feedback Tabs
# -------------------------------
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
A: You can ask in plain English! For best results, include both the nutrient and the food item. For example:  
- "What are the calories in an apple?"  
- "How much protein does tofu have?"  
- "What is the nutritional density of cream cheese?"

**Q: Any tips for better results?**  
A: Since NutrionBot is designed for single food items, please type the common name of the food (e.g., "apple", "cream cheese low fat", or "tofu").
"""

def submit_feedback(feedback):
    print("User feedback:", feedback)
    return "Thank you for your feedback!"

with gr.Blocks() as demo:
    gr.Markdown("# NutrionBot")
    with gr.Tabs():
        # Chat Tab
        with gr.Tab("Chat"):
            gr.Markdown(chat_tab_greeting)
            chat_input = gr.Textbox(lines=2, label="Your Question")
            chat_image = gr.Image(label="Upload an image (optional)", type="pil")
            # Set the submit button variant to "primary" for an orange colour
            chat_button = gr.Button("Submit", variant="primary")
            chat_output = gr.Markdown(label="Response")  # Using Markdown to render tables nicely.
            chat_button.click(fn=chatbot_interface, inputs=[chat_input, chat_image], outputs=[chat_output])
        
        # FAQ Tab
        with gr.Tab("FAQ"):
            gr.Markdown(faq_text)
        
        # Feedback Tab
        with gr.Tab("Feedback"):
            feedback_input = gr.Textbox(lines=4, label="Your Feedback")
            feedback_output = gr.Textbox(label="Feedback Response")
            feedback_button = gr.Button("Submit Feedback", variant="primary")
            feedback_button.click(fn=submit_feedback, inputs=[feedback_input], outputs=[feedback_output])
            
    demo.launch(debug=True, share=True)



