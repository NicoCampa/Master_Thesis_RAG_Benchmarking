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
    
    qa_prompt = f"""
    Based on the following company profile, generate exactly {QUESTIONS_PER_COMPANY} creative and insightful question-answer pairs.
    
    COMPANY PROFILE:
    {company_profile}
    
    INSTRUCTIONS:
    1. Create {QUESTIONS_PER_COMPANY} diverse and interesting questions that reveal unique aspects of the company.
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
    5. Format your response as a JSON array with "question" and "ground_truth" fields
    
    Example format:
    [
        {{"question": "How did Company X's unconventional approach to employee benefits lead to their breakthrough in market share?", 
          "ground_truth": "Their 20-hour workweek policy led to 40% higher productivity and a 15% market share increase in 2022"}},
        {{"question": "What unexpected synergy emerged from their partnership with Company Y in the sustainability sector?", 
          "ground_truth": "The partnership led to a revolutionary recycling process that reduced manufacturing costs by 35% while eliminating 90% of waste"}}
    ]
    
    Return only the valid JSON array with exactly {QUESTIONS_PER_COMPANY} creative questions, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a business analyst creating insightful questions about company strategies and innovations."},
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
        
        # Add metadata to each QA pair
        for pair in qa_pairs:
            pair["sector"] = sector
            pair["company_index"] = company_index
        
        return qa_pairs
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return [{"question": f"What makes this {sector} company unique?", 
                "ground_truth": "Error generating specific question",
                "sector": sector,
                "company_index": company_index} for _ in range(QUESTIONS_PER_COMPANY)]

def main():
    print("Starting question generation...")
    
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
    
    # Save QA pairs to CSV with metadata
    with open(QA_FILE, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["question", "ground_truth", "sector", "company_index"])
        for pair in all_qa_pairs:
            writer.writerow([
                pair["question"],
                pair["ground_truth"],
                pair["sector"],
                pair["company_index"]
            ])
    
    print(f"QA pairs saved to {QA_FILE}")
    print(f"Generated {len(all_qa_pairs)} QA pairs across {len(companies_data)} sectors")
    print("Question generation complete!")

if __name__ == "__main__":
    main() 