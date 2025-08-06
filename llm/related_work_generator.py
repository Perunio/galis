from dotenv import load_dotenv
import os
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(indent=4, sort_keys=True),
    ]
)
logger = structlog.get_logger()

load_dotenv()

PROMPT_TEXT = """
You are a research assistant AI specializing in academic writing. Your task is to generate a "Related Work" section 
for a research paper. You will be given a list of citations.

Your goal is to synthesize the provided citations into a coherent and well-structured "Related Work" section that 
contextualizes the user's project within the existing academic literature.

**PROVIDED CITATIONS:**
{citations}

**INSTRUCTIONS:**

1.  **Thematic Organization:** Do not simply list summaries of the papers. Group the provided citations into thematic 
categories based on shared concepts, methodologies, or research problems. For example, you could create categories like 
"Transformer-based Language Models," "Sentiment Analysis Techniques," and "Efficient Models for NLP." Introduce each 
theme before discussing the relevant papers.

2.  **Synthesis and Analysis:** For each thematic group, synthesize the key contributions and findings of the papers. 
Go beyond summarization; compare and contrast the different approaches. For instance, you could discuss the evolution 
of certain methods or the trade-offs between different models (e.g., accuracy vs. computational efficiency).

3.  **Identify Research Gaps:** Critically analyze the literature you are reviewing. Explicitly identify the 
limitations, open questions, or research gaps that the cited works leave unresolved. This will set the stage for 
introducing the project's contribution.

4.  **Contextualize the User's Project:** After discussing a thematic group of papers and identifying a gap, clearly 
and explicitly state how the user's project (described above) addresses this gap or builds upon the existing work. Use 
phrases like: "While these methods have shown great success, they struggle with...", "To address this limitation, our 
work introduces...", or "Building upon the foundation laid by [Author, Year], we propose a novel approach that...".

5.  **Academic Tone and Flow:** Maintain a formal, objective, and academic tone throughout the text. Ensure smooth 
transitions between paragraphs and ideas to create a coherent narrative that logically leads the reader to understand 
the novelty and importance of the user's project.

6.  **Output Format:** Generate only the text for the "Related Work" section. Do not include headers like 
"INSTRUCTIONS" or "PROVIDED CITATIONS" in the final output. The entire response should be the section text itself.
"""


def check_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set")
        return False
    logger.info(f"Gemini API Key is loaded: {api_key[:10]}...")
    return True


def create_related_work_pipeline():
    """Creates a ready-to-use pipeline for generating the Related Work section."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        input_variables=["citations"],
        template=PROMPT_TEXT
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


def generate_related_work(citations_text: str) -> str:
    """
    Main function - pass citations, get Related Work

    Args:
        citations_text: Text with citations (can be a list or a string)

    Returns:
        The generated Related Work section
    """
    pipeline = create_related_work_pipeline()
    result = pipeline.invoke({"citations": citations_text})
    return result


if __name__ == "__main__":

    my_citations = """
Top 5 Citation Predictions:
  - Title: 'deterministic construction of rip matrices in compressed sensing from constant weight codes'
  - Title: 'mizar items exploring fine grained dependencies in the mizar mathematical library'
  - Title: 'rateless lossy compression via the extremes'
  - Title: 'towards autonomic service provisioning systems'
  - Title: 'anonymization with worst case distribution based background knowledge'
    """

    print("Generuję Related Work...")
    print("=" * 50)

    try:
        related_work = generate_related_work(my_citations)
        print(related_work)
    except Exception as e:
        print(f"Błąd: {e}")
        print("\n=== INSTRUKCJE KONFIGURACJI ===")
        print("1. Stwórz plik .env w tym samym folderze co skrypt")
        print("2. Dodaj do niego linię: GOOGLE_API_KEY=twój_klucz")
        print("3. Uzyskaj klucz na: https://makersuite.google.com/app/apikey")
        check_api_key()