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
You are a research assistant specializing in academic writing. Your task is to generate a "Related Work" section 
for a research paper. You will be given paper's title, abstract and a list of citations.

Your goal is to synthesize the provided citations into a coherent and well-structured "Related Work" section that 
contextualizes the user's project within the existing academic literature.

**PROVIDED TITLE**
{title}

**PROVIDED ABSTRACT**
{abstract}**

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

6.  **Domain Sensitivity:** Adapt the discussion to the specific research domain indicated by the title and abstract. 
Use appropriate terminology and focus on concepts, methods, and challenges relevant to that particular field of study.

7.  **Output Format:** Generate only the text for the "Related Work" section. Do not include headers like 
"INSTRUCTIONS" "PAPER TITLE", "RELATED WORK" or "PROVIDED CITATIONS" in the final output. Do not use markdown syntax.
The entire response should be the section text itself, ready to be inserted into an academic paper.
"""


def check_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set")
        return False
    logger.info(f"Gemini API Key is loaded: {api_key[:10]}...")
    return True


def create_related_work_pipeline():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["title", "abstract", "citations"], template=PROMPT_TEXT
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


def generate_related_work(
    pipeline, title: str, abstract: str, citations_text: str
) -> str:
    result = pipeline.invoke(
        {"title": title, "abstract": abstract, "citations": citations_text}
    )
    return result


if __name__ == "__main__":
    title = "Privacy-Preserving Data Analysis in Distributed Systems: A Comprehensive Framework"

    abstract = """
    This paper presents a novel framework for privacy-preserving data analysis in distributed computing environments. 
    We propose a hybrid approach that combines differential privacy mechanisms with secure multi-party computation 
    to enable statistical analysis while maintaining strong privacy guarantees. Our framework addresses key challenges 
    in distributed data processing, including data heterogeneity, communication overhead, and scalability constraints. 
    Through extensive experiments on real-world datasets, we demonstrate that our approach achieves comparable accuracy 
    to centralized methods while providing provable privacy protection. The proposed system shows significant improvements 
    in computational efficiency compared to existing privacy-preserving solutions, making it practical for large-scale 
    deployment in enterprise environments.
    """

    citations = """
Top 5 Citation Predictions:
  - Title: 'deterministic construction of rip matrices in compressed sensing from constant weight codes'
  - Title: 'mizar items exploring fine grained dependencies in the mizar mathematical library'
  - Title: 'rateless lossy compression via the extremes'
  - Title: 'towards autonomic service provisioning systems'
  - Title: 'anonymization with worst case distribution based background knowledge'
    """

    print("Generating Related Work...")
    print("-" * 50)

    try:
        pipeline = create_related_work_pipeline()
        related_work = generate_related_work(pipeline, title, abstract, citations)
        print(related_work)
    except Exception as e:
        print(f"Error: {e}")
        print("1. Create a .env file in the same folder as the script")
        print("2. Add the line: GOOGLE_API_KEY=your_key")
        print("3. Get the key at: https://makersuite.google.com/app/apikey")
        check_api_key()
