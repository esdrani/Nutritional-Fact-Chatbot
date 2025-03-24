# %%
###########################################
# Complete End-to-End Nutrition Chatbot
###########################################

# -------------------------------
# 1. Import Modules
# -------------------------------
import os
import re
import pandas as pd
import gradio as gr
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# -------------------------------
# 2. Set Up OpenAI API Key Securely
# -------------------------------
import os
import openai

# Remove or comment out the line setting the key directly
# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

openai.api_key = os.getenv("OPENAI_API_KEY")


# -------------------------------
# 3. Data Preprocessing (Segment 1)
# -------------------------------
# Define your dataset folder (use the absolute path)
dataset_folder = "Dataset"

# List of your five CSV files
csv_files = [
    os.path.join(dataset_folder, "FOOD-DATA-GROUP1.csv"),
    os.path.join(dataset_folder, "FOOD-DATA-GROUP2.csv"),
    os.path.join(dataset_folder, "FOOD-DATA-GROUP3.csv"),
    os.path.join(dataset_folder, "FOOD-DATA-GROUP4.csv"),
    os.path.join(dataset_folder, "FOOD-DATA-GROUP5.csv")
]

# Load and merge the CSV files
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"Loaded {file} successfully.")
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        
data = pd.concat(dfs, ignore_index=True)
print("Merged dataset shape before cleaning:", data.shape)

# Remove extra columns (headers starting with 'Unnamed')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
print("Dataset shape after removing extra columns:", data.shape)

# Standardize the 'food' column (make lowercase and trim spaces)
if 'food' in data.columns:
    data['food'] = data['food'].str.lower().str.strip()
else:
    print("Warning: 'food' column not found in dataset!")

# Remove duplicate food entries
data = data.drop_duplicates(subset=['food'])
print("Dataset shape after removing duplicates:", data.shape)

# Convert all columns (except 'food') to numeric and fill missing values with 0
numeric_cols = data.columns.drop('food')
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data[numeric_cols] = data[numeric_cols].fillna(0)

# (Optional) Print missing values and descriptive statistics for verification
print("Missing values per column:")
print(data.isnull().sum())
print("Descriptive statistics for numeric columns:")
print(data[numeric_cols].describe())

# Build a lookup dictionary: key = food name, value = nutritional info (as a dictionary)
food_dict = {row['food']: row.drop('food').to_dict() for _, row in data.iterrows()}
example_food = list(food_dict.keys())[0]
print(f"Example food entry for '{example_food}':")
print(food_dict[example_food])

# Save the cleaned dataset for future use
output_path = os.path.join(dataset_folder, "FOOD-DATA-MERGED_CLEANED.csv")
data.to_csv(output_path, index=False)
print(f"Cleaned dataset saved as '{output_path}'.")

# -------------------------------
# 4. Chatbot Query Processing & IR-based Retrieval (Segment 2)
# -------------------------------
def retrieve_results(query, top_n=3):
    """
    Builds a mini-corpus (food name plus nutritional info) and uses TF-IDF with cosine similarity
    to retrieve the top matching food entries.
    """
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
# 5. Retrieval-Augmented Generation (RAG) with OpenAI (Segment 3)
# -------------------------------
def generate_conversational_answer(query, context):
    """
    Uses OpenAI's GPT-4-0314 to generate a polite, detailed answer based on the query and provided nutritional context.
    """
    try:
        prompt = (
            f"Here is the nutritional data:\n{context}\n"
            f"Please provide a detailed, polite answer to the following question:\n{query}"
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

def generate_answer(query):
    """
    Processes the user's query to identify the nutrient keyword, 
    attempts substring-based partial matches for the food name, 
    and if that fails, uses TF-IDF-based retrieval.
    Then returns a final GPT-4-0314 augmented answer.
    """
    query_lower = query.lower()
    
    # 1. Identify nutrient keyword (if any)
    nutrient_found = None
    nutrient_keywords = [
        "calories", "fat", "protein", "carbohydrates", "sugars",
        "fiber", "cholesterol", "sodium", "vitamin", "mineral"
    ]
    for kw in nutrient_keywords:
        if kw in query_lower:
            nutrient_found = kw
            break

    # 2. Attempt partial substring matches for the food name
    partial_matches = []
    for food_name in food_dict.keys():
        if food_name in query_lower or query_lower in food_name:
            partial_matches.append(food_name)

    if len(partial_matches) == 1:
        # Exactly one partial match -> pick it
        chosen_food = partial_matches[0]
        print(f"[DEBUG] Substring-based match found: {chosen_food}")
        food_item = chosen_food
    elif len(partial_matches) > 1:
        # Multiple partial matches -> ask user to specify
        matches_str = ", ".join(partial_matches[:10])  # show up to 10 matches
        return (None, f"I found multiple foods containing that name: {matches_str}. Please specify which one you mean.")
    else:
        # 3. No partial substring match found -> fallback to TF-IDF retrieval
        retrieved = retrieve_results(query, top_n=3)
        if not retrieved:
            return (None, "I couldn't find any matching foods. Please try a different query.")

        # Sort by similarity
        retrieved = sorted(retrieved, key=lambda x: x["similarity"], reverse=True)
        top1 = retrieved[0]
        sim1 = top1["similarity"]
        sim2 = retrieved[1]["similarity"] if len(retrieved) > 1 else 0.0

        # Confidence thresholds
        confidence_threshold = 0.2
        separation_threshold = 0.05

        # If the top match is below confidence_threshold, indicate uncertainty
        if sim1 < confidence_threshold:
            return (None, "I'm not sure which food you mean. Could you be more specific?")

        # If the top two matches are too close, ask the user to clarify
        if (sim1 - sim2) < separation_threshold and len(retrieved) > 1:
            candidates = [item["FoodName"] for item in retrieved]
            return (None, f"I found multiple possible matches: {', '.join(candidates)}. Please specify which one you meant.")

        # Use the best match
        food_item = top1["FoodName"]

    # 4. Retrieve the food info from the dictionary
    info = food_dict.get(food_item)
    if not info:
        return (None, f"Found '{food_item}' but no nutritional info is available, sorry.")

    # 5. Build a basic answer (direct nutrient lookup or summary)
    if nutrient_found:
        nutrient_value = None
        for k in info.keys():
            if nutrient_found in k.lower():
                nutrient_value = info[k]
                break
        if nutrient_value is not None:
            basic_answer = f"The {nutrient_found} content of {food_item} is {nutrient_value} per serving."
        else:
            basic_answer = f"Sorry, I could not find information on {nutrient_found} for {food_item}."
    else:
        summary_keys = ["calories", "fat", "protein", "carbohydrates"]
        details = []
        for s in summary_keys:
            for k in info.keys():
                if s in k.lower():
                    details.append(f"{s}: {info[k]}")
                    break
        basic_answer = f"Here is the nutritional info for {food_item}: {', '.join(details)}."

    # 6. Use RAG approach with GPT for a detailed answer
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
# 6. Image Classification Placeholder (Segment 4)
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
        # Process the image using a pre-trained model
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
# 7. Gradio Chatbot Interface (Segment 5)
# -------------------------------
def chatbot_interface(user_input, user_image):
    """
    Gradio interface for the chatbot.
    If an image is uploaded, it prints the image size (placeholder for future integration).
    Then it processes the text query and returns the answer.
    """
    try:
        # Process image if provided (currently, only prints its size)
        if user_image is not None:
            print("User uploaded an image of size:", user_image.size)
            # Uncomment below to integrate image classification:
            # recognized_class = classify_image(user_image)
            # user_input += f" recognized as: {recognized_class}"
        
        # Process the text query and generate an answer
        _, answer = generate_answer(user_input)
        return answer

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error occurred: {e}"

# -------------------------------
# 8. Build and Launch the Gradio Interface
# -------------------------------
iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(lines=2, label="Your Question"),
        gr.Image(label="Upload an image (optional)", type="pil")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Enhanced Food Nutrition Chatbot",
    description="Ask nutrition questions about food, or upload an image placeholder!"
)

# Launch the interface. The Gradio app will display a local URL and generate a public share link.
iface.launch(debug=True, share=True)




# %%
