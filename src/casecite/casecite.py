"""
LegalCiteCheck: A Legal Citation Research System
This script defines a LegalCiteCheck class that uses a LangChain model to perform multi-step legal citation research.
"""
import json
import logging
import os
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
import re

from casecite.model_params import ModelParams

# Define output models for structured parsing
class Citation(BaseModel):
    """A legal citation with verification status."""
    citation_text: str = Field(description="The full citation text in Bluebook format")
    description: str = Field(description="Brief description of what the citation contains")
    relevance: str = Field(description="How this citation relates to the proposition")

class VerifiedCitation(BaseModel):
    """A verified legal citation with confidence and details."""
    citation_text: str = Field(description="The full citation text in Bluebook format")
    confidence: str = Field(description="Confidence level: High, Medium, Low, or UNABLE TO VERIFY")
    holding: str = Field(description="The specific holding or statutory language")
    flag: str = Field(description="Whether this citation supports or refutes the proposition")
    analysis: str = Field(description="How this directly supports or refutes the proposition")
    verification: str = Field(description="Whether this citation is verified (High or Medium confidence)")
    is_verified: bool = Field(description="Whether this citation is verified")

class CitationResult(BaseModel):
    """The final result with verified citations and limitations."""
    verified_citations: List[VerifiedCitation] = Field(description="List of verified citations")
    limitations: str = Field(description="Limitations of the research")
    conclusion: str = Field(description="Conclusion of the research")

# LangChain Citation Research System
class LegalCitationResearcher:
    def __init__(self, config: ModelParams):
        """Initialize the researcher with the specified model and prompts."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if config.vendor == "anthropic":
            self.llm = ChatAnthropic(
                model=config.model,
                anthropic_api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        elif config.vendor == "openai":
            self.llm = ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        elif config.vendor == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=f'models/{config.model}',
                google_api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        elif config.vendor == 'groq':
            self.llm = ChatGroq(
                model=config.model,
                groq_api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        
        # Set up the system prompts
        self.citation_system_prompt = """
        You are LegalCiteCheck, a specialized legal research assistant. Your primary function is to provide accurate, 
        verifiable legal citations for Texas family law, Texas Rules of Evidence, Texas Rules of Civil Procedure, and Texas Civil Practice and Remedies Code.
        
        IMPORTANT RULES:
        1. Only list citations that might be relevant to the proposition
        2. Provide each citation in proper Bluebook format
        3. Include a brief description of what each citation contains
        4. Focus on Texas family law authorities unless specifically directed otherwise
        5. Always output json content between `<json_output>` and `</json_output>` tags
        """

        self.citation_user_prompt = """
                Create a comprehensive list of Texas family law citations that might address this proposition:
                
                "{proposition}"
                
                For each citation:
                1. Provide the full citation in proper Bluebook format
                2. Include a brief description of what the citation contains
                3. Explain why it might be relevant to the proposition
                
                Format the citations as a json list of objects with the following structure between <json_output> tags:
                
                ```json
                [
                    {{
                        "citation_text": "Tex. Fam. Code § 6.002",
                        "description": "Placeholder description.",
                        "relevance": "Placeholder relevance."
                    }},
                    {{
                        "citation_text": "Placeholder citation",
                        "description": "Placeholder description.",
                        "relevance": "Placeholder relevance."
                    }}
                ]
                ```

                **Important:** Place your final response within `<json_output>` and `</json_output>` tags, as shown below:

                ```text
                <json_output>
                [JSON content here]
                </json_output>
                ```
                """
        
        self.verification_system_prompt = """
        You are LegalCiteCheck Verification Specialist. Your job is to critically examine legal citations
        and determine if they actually support or refute a given proposition.
        
        IMPORTANT RULES:
        1. Examine each citation carefully and recall specific details
        2. Be honest about your confidence level for each citation
        3. Only mark citations as verified if you have High or Medium confidence
        4. Provide the exact holding or statutory language that makes the citation relevant
        5. Be transparent about any limitations in your knowledge
        6. Always output json content between `<json_output>` and `</json_output>` tags
        """

        self.verification_user_prompt = """
                For each of these Texas family law citations, verify if it directly supports or refutes the follwing proposition:
                
                "{proposition}"
                
                Citations to verify:
                {citations}
                
                For each citation:
                1. Recall the specific holding or statutory language. If you cannot recall, Flag as "UNABLE TO VERIFY"
                2. Rate your confidence in your recall (High/Medium/Low)
                3. If the the specific holding or statutory language directly addresses the proposition, include a Flag saying "SUPPORTS" or "REFUTES" the proposition
                4. If the the specific holding or statutory language does NOT directly address the proposition, Flag it as "NOT RELEVANT"
                5. Explain exactly how it supports or refutes the proposition
                6. Mark any citations you cannot verify with specific details as "UNABLE TO VERIFY" in the "verification" field
                
                Format your response as json content containing a list of objects, in the following json structure:
                ```json
                [
                    {{
                        "citation_text": "Tex. Fam. Code § 6.002",
                        "holding": "Placeholder holding or statutory language that makes this citation relevant.",
                        "confidence": "Low",
                        "flag": "SUPPORTS",
                        "analysis": "Placeholder explanation of how this citation supports or refutes the proposition.",
                        "verification": "Placeholder verification and explanation.",
                        "is_verified": false
                    }},
                    {{
                        "citation_text": "Placeholder citation",
                        "holding": "Placeholder holding or statutory language.",
                        "confidence": "High",
                        "flag": "UNABLE TO VERIFY",
                        "analysis": "Placeholder explanation of how this citation supports or refutes the proposition.",
                        "verification": "Placeholder verification and explanation.",
                        "is_verified": true
                    }}
                ]
                ```

                **Important:** Place your final response within `<json_output>` and `</json_output>` tags, as shown below:

                ```text
                <json_output>
                [JSON content here]
                </json_output>
                ```
            """
        
        self.conclusion_system_prompt = """
        You are LegalCiteCheck Conclusion Analyst. Your role is to synthesize the verified citations
        and provide a concise conclusion based on the evidence.

        IMPORTANT RULES:
        1. Consider the verified citations as evidence
        2. Formulate a conclusion based on the verified citations
        3. Be clear and concise in your final statement
        """

        self.conclusion_user_prompt = """
                <research>
                {research_results}
                </research>

                Based on the above research related to this proposition:
                
                "{proposition}"
                
                Please provide a concise conclusion based on the research cited above.
                """
        
        self.limitations_user_prompt = """
                Based on the verification process for citations related to this proposition:
                
                "{proposition}"
                
                Please provide a concise paragraph describing the limitations of your research, such as:
                - Recent developments after your training data
                - Potentially relevant authority you cannot confidently recall
                - Areas where additional research would be beneficial
                
                Be honest about any gaps in your knowledge that might affect the completeness of the citation list.
                """
        
    def extract_citations(self, text: str) -> List[Citation]:
        """Extract citations from the model's response."""
        self.logger.info(f"Extracting citations from text: %s", text)

        if isinstance(text, dict):
            text = json.dumps(text, indent=4)
        else:
            text = self.extract_text(text)

        citations = json.loads(text)
        return [Citation(**cite) for cite in citations]
    
    def extract_verified_citations(self, text: str) -> List[VerifiedCitation]:
        """Extract verified citations from the model's response."""
        self.logger.info(f"Extracting verified citations from text: %s", text)

        if isinstance(text, dict):
            text = json.dumps(text, indent=4)
        else:
            text = self.extract_text(text)

        citations = json.loads(text)
        return [VerifiedCitation(**cite) for cite in citations]
    
    def extract_limitations(self, text: str) -> str:
        """Extract limitations from the model's response."""
        if isinstance(text, dict):
            json_text = json.dumps(text, indent=4)
        else:
            json_text = self.extract_text(text)
        
        self.logger.warning(f"Extracting limitations from text: %s", json_text)

        json_object = json.loads(json_text)
        limitations_text = json_object.get('limitations_statement', f"*{text}")
        self.logger.warning(f"Extracted limitations: %s", limitations_text)

        return limitations_text.strip()
    
    def extract_text(self, text: str) -> str:
        """Extract text from the model's response."""

        # The desired text will either be between<json_output> tags or "```json" and "```"
        # Worse, sometimes the LLM includes both sets of tags. In which case, we need
        # to extract the text from the <json_output> tags.
        if "```json" in text:
            text = re.search(r'.*```json(.*?)```.*', text, re.DOTALL).group(1)
        elif "<json_output>" in text:
            text = re.search(r'.*\<json_output\>(.*?)\</json_output\>.*', text, re.DOTALL).group(1)
        return text
    
    def get_initial_citations(self, proposition: str) -> List[Citation]:
        """Step 1: Generate initial citation list."""
        citation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.citation_system_prompt),
            HumanMessagePromptTemplate.from_template(template=self.citation_user_prompt)
        ])
        
        citation_chain = LLMChain(
            llm=self.llm,
            prompt=citation_prompt,
            verbose=True
        )
        
        response = citation_chain.run(proposition=proposition)
        return self.extract_citations(response)
    
    def verify_citations(self, citations: List[Citation], proposition: str) -> List[VerifiedCitation]:
        """Step 2: Verify each citation."""
        verification_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.verification_system_prompt),
            HumanMessagePromptTemplate.from_template(template=self.verification_user_prompt)
        ])

        # Format citations for the prompt
        citations_text = "\n".join([f"{i+1}. {c.citation_text} - {c.description}" 
                                   for i, c in enumerate(citations)])
        
        verification_chain = LLMChain(
            llm=self.llm,
            prompt=verification_prompt,
            verbose=True
        )
        
        response = verification_chain.run(proposition=proposition, citations=citations_text)

        self.logger.info(f"Verification response: %s", response)
        
        # Process the verification results
        verified_citations = self.extract_verified_citations(response)

        for i, citation in enumerate(verified_citations, 1):
            if citation.confidence in ["High", "Medium"]:
                verified_citations[i-1].is_verified = True
            else:
                verified_citations[i-1].is_verified = False

        return verified_citations
    
    def draw_conclusion(self, proposition: str, verified_citations: List[VerifiedCitation]) -> str:
        conclusion_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.conclusion_system_prompt),
            HumanMessagePromptTemplate.from_template(template=self.conclusion_user_prompt)
        ])

        # Format the research results for the prompt
        research_results = self.plain_text_report(verified_citations, proposition)

        conclusion_chain = LLMChain(
            llm=self.llm,
            prompt=conclusion_prompt,
            verbose=True
        )

        response = conclusion_chain.run(proposition=proposition, research_results=research_results)
        return response
    
    def create_final_report(self, verified_citations: List[VerifiedCitation], proposition: str, conclusion: str) -> CitationResult:
        """Step 3: Create a final report with only verified citations."""
        sorted_citations = self.sort_verified_citations(verified_citations)
        verified_citations.sort(key=lambda x: (x.flag, x.confidence, not x.is_verified), reverse=True)
        
        # Filter to only include verified citations
        final_citations = [c for c in verified_citations if c.is_verified]
        
        # Use the LLM to generate limitations text
        limitations_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.verification_system_prompt),
            HumanMessagePromptTemplate.from_template(template=self.limitations_user_prompt)
        ])
        
        limitations_chain = LLMChain(
            llm=self.llm,
            prompt=limitations_prompt,
            verbose=True
        )
        
        limitations = limitations_chain.run(proposition=proposition)
        limitations_statement = self.extract_limitations(limitations)
        self.logger.info(f"Limitations statement: %s", limitations_statement)
        
        return CitationResult(
            verified_citations=sorted_citations,  #final_citations,
            limitations=limitations_statement,
            conclusion=conclusion
        )
    
    def sort_verified_citations(self, verified_citations: List[VerifiedCitation]) -> List[VerifiedCitation]:
        """Sort verified citations by verification status, flag, and confidence."""
        flag_order = {"SUPPORTS": 0, "REFUTES": 1, "NOT RELEVANT": 2, "UNABLE TO VERIFY": 3}
        confidence_order = {"High": 0, "Medium": 1, "Low": 2}
        is_verified_order = {True: 0, False: 1}

        verified_citations.sort(key=lambda x: (is_verified_order.get(x.is_verified, 999), flag_order.get(x.flag, 999), confidence_order.get(x.confidence, 999)))

        return verified_citations
    
    def research_legal_proposition(self, proposition: str, user_key: str = None) -> Tuple[CitationResult, str]:
        """
        Complete multi-step legal research process.

        Args:
        - proposition (str): The legal proposition to research.
        - user_key (str): A unique user key to track the research process.

        Returns:
        - result (CitationResult): The final citation result.
        - user_key (str): The unique user key.
        """
        # Step 1: Generate initial citations
        initial_citations = self.get_initial_citations(proposition)
        
        # Step 2: Verify each citation
        verified_citations = self.verify_citations(initial_citations, proposition)

        # Step 3: Create summary analysis of proposition vs citations using the text report
        conclusion = self.draw_conclusion(proposition, verified_citations)
        
        # Step 4: Create final report
        result = self.create_final_report(verified_citations, proposition, conclusion)
        
        return result, user_key
    
    def markdown_report(self, result: CitationResult, proposition: str) -> str:
        """Generate a markdown report from the final citation result."""
        report = "# CITATION REPORT\n"
        report += f"\n> Proposition: {proposition}"
        report += "\n\n# RESEARCHED CITATIONS"
        for i, citation in enumerate(result.verified_citations, 1):
            report += f"\n\n## {i}. {citation.citation_text}"
            report += f"\n\n**Confidence**: {citation.confidence}"
            report += f"\n\n**Holding**: {citation.holding}"
            report += f"\n\n**{citation.flag.title()}**: {citation.analysis}"
            report += f"\n\n**Verified**: {citation.verification}"
        
        report += "\n\n# CONCLUSION"
        report += f"\n\n{result.conclusion}"
        report += "\n\n# RESEARCH LIMITATIONS"
        report += f"\n\n{result.limitations}"
        
        return report
    
    def plain_text_report(self, citations: list, proposition: str) -> str:
        """Generate a plain text report from the final citation result."""
        report = "RESEARCHED CITATIONS\n"
        for i, citation in enumerate(citations, 1):
            report += f"\n{i}. {citation.citation_text}\n"
            report += f"Confidence: {citation.confidence}\n"
            report += f"Holding: {citation.holding}\n"
            report += f"Relevance: {citation.analysis}\n"
            report += f"Verified: {citation.verification}\n"
            report += f"Flag: {citation.flag}\n"
        
        return report

# Example usage
def main():
    model_params = ModelParams()
    researcher = LegalCitationResearcher(model_params)
    
    # Example proposition
    proposition = input("Enter a legal proposition: ")
    if not proposition:
        proposition = "In Texas, separate property can be transformed into community property through commingling."
    
    result, user_key = researcher.research_legal_proposition(proposition, user_key)
    markdown_report = researcher.markdown_report(result, proposition)
    print(markdown_report)

if __name__ == "__main__":
    main()
