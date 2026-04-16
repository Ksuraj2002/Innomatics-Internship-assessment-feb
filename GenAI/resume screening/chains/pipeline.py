from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from prompts.templates import EXTRACTION_PROMPT, SCORING_PROMPT

def build_screening_pipeline(model_name="gpt-3.5-turbo", temperature=0):
    # Initialize the LLM and the JSON parser
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    parser = JsonOutputParser()

    # Define the individual chains
    extraction_chain = EXTRACTION_PROMPT | llm | parser
    scoring_chain = SCORING_PROMPT | llm | parser

    # Build the full LCEL Pipeline
    # Passes the extracted data AND the original job description into the scoring chain
    pipeline = (
        {
            "extracted_data": {"resume": itemgetter("resume")} | extraction_chain,
            "job_description": itemgetter("job_description")
        }
        | scoring_chain
    )

    return pipeline