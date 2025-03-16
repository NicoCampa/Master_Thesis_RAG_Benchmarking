import os
import csv
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration
OUTPUT_DIR = "./data/Ragas"
COMPANIES_FILE = f"{OUTPUT_DIR}/companies.json"
QA_FILE = f"{OUTPUT_DIR}/qa.csv"
QUESTIONS_PER_COMPANY = 3

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_qa_pairs(company_profile, sector, company_index):
    """Generate creative Q&A pairs based on a company profile"""
    print(f"Generating Q&A pairs for {sector} company #{company_index}...")
    
    # Generate 1 of each type by default (3 total)
    specific_count = 1
    general_count = 1
    multihop_count = 1
    
    qa_prompt = f"""
    Based on the following company profile, generate exactly {QUESTIONS_PER_COMPANY} creative and insightful question-answer pairs.
    
    COMPANY PROFILE:
    {company_profile}
    
    INSTRUCTIONS:
    1. Create a MIX of {QUESTIONS_PER_COMPANY} diverse questions with EXACTLY:
       - {specific_count} SPECIFIC question: Focused on precise details, statistics, or factual information
       - {general_count} GENERAL question: About broader strategies, approaches, or company philosophy
       - {multihop_count} MULTI-HOP question: Requires connecting multiple facts from different parts of the profile
    
    2. AVOID basic questions like "When was the company founded?" or "Who is the CEO?"
    
    3. Instead, focus on:
       - Strategic decisions and their impacts
       - Innovative products and their unique features
       - Unusual company practices or traditions
       - Complex relationships between different aspects of the business
       - Surprising facts or unexpected developments
       - Environmental or social impact initiatives
       - Unique market approaches or competitive strategies
       
    4. Questions should have specific, verifiable answers from the text
    
    5. Clearly mark each question type at the beginning:
       - SPECIFIC: [S] Your specific question here?
       - GENERAL: [G] Your general question here?
       - MULTI-HOP: [M] Your multi-hop question here?
    
    6. Format your response as a JSON array with "question", "ground_truth", and "question_type" fields
    
    Example format:
    [
        {{
            "question": "[S] How did Company X's acquisition of TechFirm impact their market position in arid regions?", 
            "ground_truth": "The acquisition led to the development of hybrid power plants capable of both energy production and freshwater generation, significantly bolstering their market position in arid regions.",
            "question_type": "specific"
        }},
        {{
            "question": "[G] How does Company X approach sustainability across its business operations?", 
            "ground_truth": "Company X approaches sustainability through renewable energy-powered operations, biodegradable product materials, and an initiative to power 100% of their operations sustainably by 2030.",
            "question_type": "general"
        }},
        {{
            "question": "[M] How did the company's early innovation in sound technology combine with their later pivot to create their most successful product line?", 
            "ground_truth": "Their early innovations in sound technology from 2010, when combined with their unexpected pivot to wave resonance in 2015, enabled the development of their most successful Resonance Extract Generator product line.",
            "question_type": "multihop"
        }}
    ]
    
    Return only the valid JSON array with exactly {QUESTIONS_PER_COMPANY} creative questions (one of each type), nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a business analyst creating diverse types of questions to thoroughly test retrieval and reasoning capabilities."},
                {"role": "user", "content": qa_prompt}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
            
        qa_pairs = json.loads(content)
        
        # Add metadata to each QA pair and clean up question format
        for pair in qa_pairs:
            pair["sector"] = sector
            pair["company_index"] = company_index
            
            # Remove the question type prefix if present
            if pair["question"].startswith("[S]") or pair["question"].startswith("[G]") or pair["question"].startswith("[M]"):
                pair["question"] = pair["question"].split("] ", 1)[1] if "] " in pair["question"] else pair["question"][4:].strip()
        
        return qa_pairs
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return [
            {"question": f"What specific metric highlights this {sector} company's success?", 
             "ground_truth": "Error generating specific question",
             "sector": sector, 
             "company_index": company_index,
             "question_type": "specific"},
            {"question": f"How does this {sector} company approach innovation?", 
             "ground_truth": "Error generating general question",
             "sector": sector, 
             "company_index": company_index,
             "question_type": "general"},
            {"question": f"How did this {sector} company's early strategy connect to their recent product success?", 
             "ground_truth": "Error generating multi-hop question",
             "sector": sector, 
             "company_index": company_index,
             "question_type": "multihop"}
        ]

def main():
    print("Starting question generation with mixed question types...")
    
    if not os.path.exists(COMPANIES_FILE):
        print(f"Error: Companies file {COMPANIES_FILE} not found.")
        print("Please run generate_context.py first.")
        return
    
    # Load the companies data
    with open(COMPANIES_FILE, "r") as f:
        companies_data = json.load(f)
    
    # Generate QA pairs for each company
    all_qa_pairs = []
    
    for sector, companies in companies_data.items():
        for company in companies:
            qa_pairs = generate_qa_pairs(
                company["profile"],
                sector,
                company["company_index"]
            )
            all_qa_pairs.extend(qa_pairs)
            time.sleep(2)
    
    # Save QA pairs to CSV with metadata including question type
    with open(QA_FILE, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["question", "ground_truth", "sector", "company_index", "question_type"])
        for pair in all_qa_pairs:
            writer.writerow([
                pair["question"],
                pair["ground_truth"],
                pair["sector"],
                pair["company_index"],
                pair.get("question_type", "unknown")  # Include question type
            ])
    
    # Count questions by type
    type_counts = {}
    for pair in all_qa_pairs:
        q_type = pair.get("question_type", "unknown")
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    print(f"QA pairs saved to {QA_FILE}")
    print(f"Generated {len(all_qa_pairs)} QA pairs across {len(companies_data)} sectors")
    print("Question breakdown by type:")
    for q_type, count in type_counts.items():
        print(f"  - {q_type.capitalize()}: {count}")
    print("Question generation complete!")

if __name__ == "__main__":
    main() 