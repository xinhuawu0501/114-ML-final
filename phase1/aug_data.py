import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch

# 1. Configuration
INPUT_FILE = "../dataset/pr/train_nostra.csv"  # Your current train file
OUTPUT_FILE = "../dataset/pr/train_augmented_nostra.csv"
TARGET_CLASS = 4  
NUM_AUGMENTATIONS = 10 # How many new rows to create per original row

# 2. Load the Augmenter (Using PubMedBERT for medical context)
print("Loading PubMedBERT augmenter...")
device = "cuda" if torch.cuda.is_available() else "cpu"
aug = naw.ContextualWordEmbsAug(
    model_path='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', 
    action="substitute",  # 'substitute' swaps words, 'insert' adds words
    device=device,
    aug_p=0.3  # Swap 30% of the words in the sentence
)

# 3. Load Data
df = pd.read_csv(INPUT_FILE)
print(f"Original shape: {df.shape}")

# 4. Filter for Minority Class
minority_df = df[df['class'] == TARGET_CLASS].copy()
majority_df = df[df['class'] != TARGET_CLASS].copy()
print(f"Found {len(minority_df)} samples for Class {TARGET_CLASS}")

# 5. Generate Augmentations
new_rows = []

print("Augmenting...")
for _, row in tqdm(minority_df.iterrows(), total=len(minority_df)):
    original_text = row['text']
    
    # Generate N variations
    # nlpaug returns a list of strings
    augmented_sentences = aug.augment(original_text, n=NUM_AUGMENTATIONS)
    
    # Depending on nlpaug version, result might be a string or list. Ensure list.
    if isinstance(augmented_sentences, str): 
        augmented_sentences = [augmented_sentences]

    for sent in augmented_sentences:
        new_rows.append({
            "text": sent,
            "class": TARGET_CLASS
        })

# 6. Combine and Save
augmented_df = pd.DataFrame(new_rows)
final_df = pd.concat([df, augmented_df], ignore_index=True)

# Shuffle to mix the new samples in
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"New shape: {final_df.shape}")
print(f"New Class {TARGET_CLASS} count: {len(final_df[final_df['class'] == TARGET_CLASS])}")

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")