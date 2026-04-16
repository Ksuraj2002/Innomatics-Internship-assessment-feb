from langchain_core.prompts import PromptTemplate

# Step 1: Skill Extraction Prompt
EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["resume"],
    template="""You are an expert technical recruiter. Extract the following details from the resume below.
    
    Rule: Do NOT assume skills not present in the resume. 
    
    Return the output strictly as a JSON object with the following keys: 
    - "skills" (list of strings)
    - "experience_years" (float or string)
    - "tools" (list of strings)

    Resume:
    {resume}
    """
)

# Step 2-4: Matching, Scoring, and Explanation Prompt
SCORING_PROMPT = PromptTemplate(
    input_variables=["extracted_data", "job_description"],
    template="""You are an AI hiring manager. Compare the candidate's extracted profile with the job description.
    
    Tasks:
    1. Match the extracted skills, tools, and experience against the job requirements.
    2. Assign a fit score from 0 to 100 based on alignment.
    3. Provide a detailed explanation justifying the score. Be objective and do not hallucinate.

    Extracted Candidate Data:
    {extracted_data}

    Job Description:
    {job_description}

    Return the output strictly as a JSON object with the following keys: 
    - "score" (integer)
    - "explanation" (string)
    """
)