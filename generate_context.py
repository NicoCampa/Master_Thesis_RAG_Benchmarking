import os
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
CONTEXT_FILE = f"{OUTPUT_DIR}/testContext.txt"
COMPANIES_FILE = f"{OUTPUT_DIR}/companies.json"
COMPANIES_PER_SECTOR = 2

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Company sectors to include
SECTORS = [
    "Technology",
    "Healthcare",
    "Finance",
    "Energy",
    "Retail",
    "Manufacturing",
    "Hospitality",
    "Agriculture",
    "Media & Entertainment",
    "Transportation & Logistics"
]

def generate_company_profile(sector, index):
    """Generate a detailed company profile for a specific sector"""
    print(f"Generating company profile #{index} for {sector} sector...")
    
    company_prompt = f"""
    Create an extremely detailed and comprehensive fictional company profile for a company in the {sector} sector.
    Make this company unique and interesting, with unexpected elements and innovative approaches.
    
    REQUIRED SECTIONS (each should be 5-6 paragraphs with specific details):
    1. Company Overview & History (include unique founding story and unexpected pivots)
    2. Product Portfolio (at least 15 innovative products with complete specifications)
    3. Financial Performance (detailed metrics and historical data, including unusual success metrics)
    4. Leadership Team (10+ executives with diverse backgrounds and unique career paths)
    5. Market Analysis & Competitive Positioning (include unconventional strategies)
    6. Global Operations & Infrastructure (interesting locations and facility designs)
    7. Research & Development Activities (include moonshot projects)
    8. Corporate Strategy & Future Outlook (include bold visions)
    9. Corporate Social Responsibility Initiatives (unique approaches to sustainability)
    10. Partnerships & Acquisitions (interesting collaborations and strategic moves)
    
    IMPORTANT GUIDELINES:
    1. Include at least 6-8 paragraphs in each section
    2. Provide at least 15 specific product details with exact specifications and pricing
    3. Include at least 30 specific numerical facts (dates, amounts, percentages, etc.)
    4. Create a detailed leadership team with at least 10 executives and their complete backgrounds
    5. Include at least 10 specific milestones with exact dates
    6. Add financial data for at least 5 years with specific figures
    7. Describe manufacturing/operations facilities with specific locations and capacities
    8. Include market share percentages and competitor analysis
    9. Discuss at least 3 major partnerships or acquisitions with details
    
    MAKE IT UNIQUE:
    - Include unexpected innovations or approaches
    - Add interesting cultural elements or company traditions
    - Include surprising pivot points or strategic decisions
    - Mention unique employee benefits or workplace practices
    - Describe innovative sustainability initiatives
    
    FORMAT:
    Start with a clear company name and make it memorable
    Use descriptive headings and subheadings to organize information clearly.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative business analyst creating unique and detailed company profiles."},
                {"role": "user", "content": company_prompt}
            ],
            temperature=0.8,  # Slightly higher temperature for more creativity
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating company profile: {e}")
        return f"**Fictional {sector} Company {index}**\n\nThis is a placeholder company profile due to API error."

def main():
    print("Starting company profile generation...")
    
    # Generate company profiles - two per sector
    all_profiles = []
    companies_data = {}  # Dictionary to store all companies by sector
    
    for sector in SECTORS:
        sector_companies = []
        for i in range(COMPANIES_PER_SECTOR):
            profile = generate_company_profile(sector, i+1)
            all_profiles.append(profile)
            sector_companies.append({
                "sector": sector,
                "company_index": i+1,
                "profile": profile
            })
            time.sleep(2)
        companies_data[sector] = sector_companies
    
    # Combine all profiles into one context document
    with open(CONTEXT_FILE, "w") as f:
        f.write("\n\n".join(all_profiles))
    
    # Save all companies data for question generation
    with open(COMPANIES_FILE, "w") as f:
        json.dump(companies_data, f, indent=2)
    
    print(f"All company profiles saved to {CONTEXT_FILE}")
    print(f"Companies data saved to {COMPANIES_FILE}")
    print(f"Generated {len(all_profiles)} company profiles ({COMPANIES_PER_SECTOR} per sector)")
    print("Company profile generation complete!")

if __name__ == "__main__":
    main()