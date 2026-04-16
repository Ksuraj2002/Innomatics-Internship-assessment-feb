import os
import json
from chains.pipeline import build_screening_pipeline
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "GenAI_Resume_Screening"

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def main():
    # 1. Inputs
    job_description = """
    Looking for a Data Scientist with 2+ years of experience.
    Must have strong skills in Python, Machine Learning, and NLP.
    Experience with PyTorch, LangChain, and SQL is highly preferred.
    """

    resumes = {
        "Strong Candidate": """
            Data Scientist with 3 years of experience.
            Skills: Python, Machine Learning, Deep Learning, NLP.
            Tools: PyTorch, LangChain, SQL, PostgreSQL, Git.
            Built an end-to-end sentiment analysis pipeline and retail sales forecasting model.
            """,
        "Average Candidate": """
            Data Analyst with 1.5 years of experience.
            Skills: Python, Data Analysis, Basic Machine Learning.
            Tools: Excel, SQL, Scikit-learn.
            Worked on data cleaning, basic regression models, and reporting.
            """,
        "Weak Candidate": """
            Frontend Developer with 2 years of experience.
            Skills: HTML, CSS, JavaScript, React.
            Tools: VS Code, Git, Figma.
            Built modern responsive web applications and managed component libraries.
            """
    }

    # 2. Initialize the LCEL Pipeline
    pipeline = build_screening_pipeline()

    # 3. Execution Loop
    for label, resume_text in resumes.items():
        print(f"\n{'='*40}")
        print(f"Evaluating: {label}")
        print(f"{'='*40}")
        
        try:
            # Using .invoke() as explicitly required by the assignment
            result = pipeline.invoke({
                "resume": resume_text,
                "job_description": job_description
            })
            
            # Print the formatted JSON output
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error during pipeline execution for {label}: {e}")

if __name__ == "__main__":
    main()