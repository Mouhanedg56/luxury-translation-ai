"""
Pydantic AI Workflow System for Luxury Fashion Translation
Uses Pydantic AI for structured, type-safe agentic workflows
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# Pydantic models for workflow data structures
class ContentSegment(BaseModel):
    """Represents a translatable content segment"""

    id: str = Field(..., description="Unique segment identifier")
    text: str = Field(..., description="Text content to translate")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Contextual information"
    )
    priority: int = Field(
        default=3, ge=1, le=5, description="Translation priority (1=highest)"
    )
    content_type: Literal["title", "description", "marketing", "technical", "legal"] = (
        "description"
    )
    requires_human_review: bool = Field(default=False)

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text content cannot be empty")
        return v.strip()


class FormatElement(BaseModel):
    """Represents a formatting element (HTML, Markdown, etc.)"""

    id: str = Field(..., description="Unique element identifier")
    tag: str = Field(..., description="HTML tag or markdown syntax")
    attributes: Dict[str, str] = Field(default_factory=dict)
    position: int = Field(..., description="Position in original content")
    length: int = Field(..., description="Length of the element")
    preserve_whitespace: bool = Field(default=False)
    is_self_closing: bool = Field(default=False)


class BrandContext(BaseModel):
    """Brand-specific context for translation"""

    brand_name: str = Field(..., description="Brand name")
    brand_voice: str = Field(..., description="Brand voice description")
    target_market: str = Field(..., description="Target market/culture")
    glossary: Dict[str, str] = Field(
        default_factory=dict, description="Brand-specific terminology"
    )
    forbidden_terms: List[str] = Field(default_factory=list)
    cultural_notes: Optional[str] = Field(None, description="Cultural adaptation notes")
    quality_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class TranslationTask(BaseModel):
    """Complete translation task definition"""

    task_id: str = Field(..., description="Unique task identifier")
    source_language: str = Field(..., min_length=2, max_length=5)
    target_language: str = Field(..., min_length=2, max_length=5)
    raw_content: Optional[str] = Field(
        None, description="Original raw content to be analyzed and segmented"
    )
    content_segments: List[ContentSegment] = Field(
        default_factory=list, description="Pre-segmented content (optional)"
    )
    format_elements: List[FormatElement] = Field(default_factory=list)
    brand_context: BrandContext
    quality_requirements: Dict[str, float] = Field(
        default_factory=lambda: {
            "accuracy": 0.9,
            "fluency": 0.9,
            "brand_consistency": 0.95,
            "format_preservation": 0.98,
        }
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TranslationResult(BaseModel):
    """Translation result with quality metrics"""

    segment_id: str
    original_text: str
    translated_text: str
    quality_scores: Optional[Dict[str, float]]
    provider_used: str
    processing_time_ms: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    requires_review: bool = False
    warnings: List[str] = Field(default_factory=list)


class WorkflowState(BaseModel):
    """Current state of the translation workflow"""

    task_id: str
    status: Literal[
        "pending",
        "analyzing",
        "translating",
        "reassembling",
        "reviewing",
        "completed",
        "failed",
    ]
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_step: str = "initialization"
    results: List[TranslationResult] = Field(default_factory=list)
    reassembled_content: Optional[str] = Field(None, description="Reassembled content with preserved formatting")
    errors: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Pydantic AI Agents
class FormatAnalysisResult(BaseModel):
    """Result from format analysis"""

    segments: List[ContentSegment]
    format_elements: List[FormatElement]


# Agent result models
class LuxuryTranslationResult(BaseModel):
    """Result from luxury translation - simplified for better validation"""

    translated_text: str
    confidence_score: Optional[float] = 0.85
    requires_review: Optional[bool] = False


class QualityAssessmentResult(BaseModel):
    """Result from quality assessment"""

    accuracy: float = Field(ge=0.0, le=1.0, description="Semantic correctness")
    fluency: float = Field(ge=0.0, le=1.0, description="Natural language flow")
    brand_consistency: float = Field(
        ge=0.0, le=1.0, description="Brand voice alignment"
    )
    cultural_appropriateness: float = Field(
        ge=0.0, le=1.0, description="Target market sensitivity"
    )
    luxury_positioning: float = Field(
        ge=0.0, le=1.0, description="Aspirational quality maintenance"
    )


# Lazy agent creation functions
def get_format_analysis_agent():
    """Create and return the format analysis agent"""
    return Agent(
        model="openai:gpt-4o",
        output_type=FormatAnalysisResult,
        system_prompt="""You are a luxury fashion content format analyzer. 
        Your task is to analyze content and extract translatable segments while preserving structure.
        
        Key responsibilities:
        1. Identify translatable text segments
        2. Preserve HTML/Markdown formatting context
        3. Prioritize segments based on luxury fashion importance
        4. Flag segments requiring human review for brand voice
        
        Always maintain the sophisticated tone expected for luxury brands.""",
        deps_type=BrandContext,
        retries=3,
    )


def get_luxury_translation_agent():
    """Create and return the luxury translation agent"""
    return Agent(
        model="openai:gpt-4o",
        output_type=LuxuryTranslationResult,
        system_prompt="""You are an expert luxury fashion translator specializing in preserving brand voice and cultural nuance.
        
        Return a JSON object with:
        - translated_text: the translation
        - confidence_score: a number between 0.0 and 1.0 (optional)
        - requires_review: true or false (optional)
        
        Focus on luxury fashion terminology and maintain sophisticated tone.""",
        deps_type=BrandContext,
        retries=3,
    )


def get_quality_assessment_agent():
    """Create and return the quality assessment agent"""
    # Use Anthropic if available, otherwise fall back to OpenAI
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        model = "anthropic:claude-3-5-sonnet-20241022"
    else:
        model = "openai:gpt-4o"

    return Agent(
        model=model,
        output_type=QualityAssessmentResult,
        system_prompt="""You are a luxury fashion translation quality assessor.
        
        Return a JSON object with numeric scores from 0.0 to 1.0 for:
        - accuracy: semantic correctness
        - fluency: natural language flow  
        - brand_consistency: brand voice alignment
        - cultural_appropriateness: target market sensitivity
        - luxury_positioning: aspirational quality maintenance
        
        Only return the JSON object with scores, no explanations.""",
        deps_type=BrandContext,
        retries=3,
    )


def get_content_reassembly_agent():
    """Create and return the content reassembly agent"""
    return Agent(
        model="openai:gpt-4o",
        output_type=str,
        system_prompt="""You are a content reassembly specialist for luxury fashion content.
        
        Your task is to reconstruct translated content while perfectly preserving:
        1. HTML structure and attributes
        2. Markdown formatting syntax
        3. Whitespace and layout
        4. Special characters and symbols
        5. Link targets and references
        
        Maintain structural integrity while ensuring translated text flows naturally.""",
        retries=3,
    )


# Helper functions for working with agents
async def translate_segment_with_agent(
    segment: ContentSegment,
    brand_context: BrandContext,
    source_lang: str,
    target_lang: str,
) -> TranslationResult:
    """Translate a single content segment using the luxury translation agent"""

    start_time = time.time()

    # Build context-aware prompt
    glossary_context = ""
    if brand_context.glossary:
        glossary_context = (
            f"Brand Glossary: {json.dumps(brand_context.glossary, indent=2)}"
        )

    prompt = f"""
    Translate this luxury fashion content from {source_lang} to {target_lang}:
    
    Brand Context:
    - Brand: {brand_context.brand_name}
    - Voice: {brand_context.brand_voice}
    - Target Market: {brand_context.target_market}
    - Quality Threshold: {brand_context.quality_threshold}
    
    {glossary_context}
    
    Content to Translate:
    Type: {segment.content_type}
    Priority: {segment.priority}
    Text: "{segment.text}"
    
    Requirements:
    1. Maintain luxury brand sophistication
    2. Use appropriate fashion terminology
    3. Ensure cultural appropriateness for {brand_context.target_market}
    4. Preserve emotional impact and aspirational quality
    5. Use brand glossary terms where applicable
    """

    # Execute translation with Pydantic AI
    translation_agent = get_luxury_translation_agent()
    logger.info(f"Translating segment {segment.id} with Pydantic AI luxury translator")
    # logger.info(f"Prompt: {prompt}")
    run_result = await translation_agent.run(prompt, deps=brand_context)

    processing_time = int((time.time() - start_time) * 1000)

    # Create structured result - Pydantic AI returns the structured output in the .output attribute
    output = run_result.output
    return TranslationResult(
        segment_id=segment.id,
        original_text=segment.text,
        translated_text=output.translated_text,
        quality_scores=None,
        provider_used="pydantic_ai_luxury_translator",
        processing_time_ms=processing_time,
        confidence_score=output.confidence_score or 0.85,
        requires_review=output.requires_review or False,
        warnings=[],
    )


async def assess_quality_with_agent(
    original: str, translation: str, brand_context: BrandContext
) -> Dict[str, float]:
    """Assess translation quality using the quality assessment agent"""

    prompt = f"""
    Assess this luxury fashion translation quality:
    
    Brand: {brand_context.brand_name}
    Brand Voice: {brand_context.brand_voice}
    Target Market: {brand_context.target_market}
    
    Original: {original}
    Translation: {translation}
    
    Evaluate on:
    1. Accuracy: Semantic correctness and meaning preservation
    2. Fluency: Natural language flow and readability
    3. Brand_consistency: Voice, tone, and terminology alignment
    4. Cultural_appropriateness: Target market sensitivity
    5. Luxury_positioning: Maintains aspirational and exclusive quality
    """

    quality_agent = get_quality_assessment_agent()
    run_result = await quality_agent.run(prompt, deps=brand_context)

    # Convert the structured output to Dict[str, float] for compatibility
    assessment = run_result.output
    return {
        "accuracy": assessment.accuracy,
        "fluency": assessment.fluency,
        "brand_consistency": assessment.brand_consistency,
        "cultural_appropriateness": assessment.cultural_appropriateness,
        "luxury_positioning": assessment.luxury_positioning,
    }


async def analyze_content_with_agent(
    content: str, brand_context: BrandContext
) -> FormatAnalysisResult:
    """Analyze content and extract translatable segments using the format analysis agent"""

    prompt = f"""
    Analyze this luxury fashion content and extract translatable segments while preserving structure:
    
    Brand: {brand_context.brand_name}
    Brand Voice: {brand_context.brand_voice}
    Target Market: {brand_context.target_market}
    
    Content to analyze:
    {content}
    
    Please:
    1. Identify translatable text segments
    2. Preserve HTML/Markdown formatting context
    3. Prioritize segments based on luxury fashion importance (1=highest priority like brand names/titles, 2=product descriptions, 3=general text)
    4. Classify content types (title, description, marketing, technical, legal)
    5. Flag segments requiring human review for brand voice consistency
    6. Extract format elements that need to be preserved
    
    Return structured data with segments and format_elements arrays.
    """

    format_agent = get_format_analysis_agent()
    run_result = await format_agent.run(prompt, deps=brand_context)
    return run_result.output


# Pydantic AI Workflow Orchestrator
class LuxuryTranslationWorkflow:
    """Main workflow orchestrator using Pydantic AI agents"""

    def __init__(self, openai_api_key: str, anthropic_api_key: Optional[str] = None):
        # Store API keys for environment setup
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

        # Set environment variables for Pydantic AI
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

        # Workflow state management
        self.active_workflows: Dict[str, WorkflowState] = {}

        # Log which models will be used
        quality_model = (
            "Anthropic Claude 3.5 Sonnet" if anthropic_api_key else "OpenAI GPT-4o"
        )
        logger.info(
            f"Initialized LuxuryTranslationWorkflow - Quality assessment will use: {quality_model}"
        )

    async def execute_translation_workflow(
        self, task: TranslationTask
    ) -> WorkflowState:
        """Execute complete translation workflow with Pydantic AI agents"""

        logger.info(f"Executing translation workflow for task {task.task_id}")

        # Initialize workflow state
        workflow_state = WorkflowState(
            task_id=task.task_id, status="analyzing", current_step="format_analysis"
        )
        self.active_workflows[task.task_id] = workflow_state

        try:
            # Step 1: Format Analysis and Segmentation
            logger.info(f"Starting format analysis for task {task.task_id}")
            workflow_state.status = "analyzing"
            workflow_state.progress = 0.1

            # If content_segments not provided, analyze raw content using format analysis agent
            logger.info(f"Content segments provided: {len(task.content_segments)}")
            if not task.content_segments and task.raw_content:
                # Use format analysis agent to extract segments from raw content
                analysis_result = await analyze_content_with_agent(
                    task.raw_content, task.brand_context
                )
                task.content_segments = analysis_result.segments

                # Update format elements if they were extracted
                if analysis_result.format_elements:
                    task.format_elements.extend(analysis_result.format_elements)

                workflow_state.progress = 0.2
                workflow_state.updated_at = datetime.utcnow()
                logger.info(
                    f"Format analysis completed: {len(task.content_segments)} segments extracted"
                )

            elif not task.content_segments:
                raise ValueError(
                    "Either raw_content or content_segments must be provided"
                )

            # Step 2: Translation
            logger.info(f"Starting translation for task {task.task_id}")
            workflow_state.status = "translating"
            workflow_state.current_step = "segment_translation"
            workflow_state.progress = 0.3

            translation_results = []
            total_segments = len(task.content_segments)

            for i, segment in enumerate(task.content_segments):
                # Translate segment using the helper function
                result = await translate_segment_with_agent(
                    segment=segment,
                    brand_context=task.brand_context,
                    source_lang=task.source_language,
                    target_lang=task.target_language,
                )
                # Quality assessment using the helper function
                quality_scores = await assess_quality_with_agent(
                    original=segment.text,
                    translation=result.translated_text,
                    brand_context=task.brand_context,
                )

                result.quality_scores = quality_scores
                translation_results.append(result)

                # Update progress
                segment_progress = (
                    (i + 1) / total_segments * 0.5
                )  # 50% of total progress
                workflow_state.progress = 0.3 + segment_progress
                workflow_state.updated_at = datetime.utcnow()

            # Step 3: Quality Gate Check
            workflow_state.current_step = "quality_assessment"
            workflow_state.progress = 0.8

            # Check if results meet quality thresholds
            quality_gate_passed = await self._quality_gate_check(
                translation_results, task.quality_requirements
            )

            if not quality_gate_passed:
                workflow_state.status = "reviewing"
                workflow_state.current_step = "human_review_required"
                # Flag segments that need review
                for result in translation_results:
                    if any(
                        score < threshold
                        for score, threshold in zip(
                            result.quality_scores.values(),
                            task.quality_requirements.values(),
                        )
                    ):
                        result.requires_review = True

            # Step 4: Content Reassembly (if format elements exist)
            if task.format_elements:
                workflow_state.current_step = "content_reassembly"
                workflow_state.progress = 0.9
                
                logger.info(f"Starting content reassembly for task {task.task_id}")
                
                # Use content reassembly agent to reconstruct formatted content
                reassembled_content = await self._reassemble_content_with_agent(
                    original_content=task.raw_content,
                    translation_results=translation_results,
                    format_elements=task.format_elements,
                    brand_context=task.brand_context
                )
                
                # Store reassembled content in workflow state for later use
                workflow_state.reassembled_content = reassembled_content
                
                logger.info(f"Content reassembly completed for task {task.task_id}")
            else:
                # For content without format elements, we'll handle simple reassembly in the API layer
                logger.info(f"No format elements found, skipping reassembly step for task {task.task_id}")

            # Step 5: Finalization
            workflow_state.results = translation_results
            workflow_state.status = "completed"
            workflow_state.progress = 1.0
            workflow_state.current_step = "completed"
            workflow_state.updated_at = datetime.now()

            logger.info(f"Workflow {task.task_id} completed successfully")

        except Exception as e:
            workflow_state.status = "failed"
            workflow_state.errors.append(str(e))
            workflow_state.updated_at = datetime.utcnow()
            logger.error(f"Workflow {task.task_id} failed: {e}")

        return workflow_state

    async def _quality_gate_check(
        self, results: List[TranslationResult], requirements: Dict[str, float]
    ) -> bool:
        """Check if translation results meet quality requirements"""

        for result in results:
            for metric, threshold in requirements.items():
                if metric in result.quality_scores:
                    if result.quality_scores[metric] < threshold:
                        return False
        return True

    async def _reassemble_content_with_agent(
        self,
        original_content: str,
        translation_results: List[TranslationResult],
        format_elements: List[FormatElement],
        brand_context: BrandContext
    ) -> str:
        """Use the content reassembly agent to reconstruct formatted content"""
        
        # Build translation mapping for the agent
        translation_mapping = {
            result.original_text: result.translated_text
            for result in translation_results
        }
        
        # Build format context information
        format_context = {
            "total_elements": len(format_elements),
            "element_types": list(set(elem.tag for elem in format_elements)),
            "has_html": any("<" in elem.tag and ">" in elem.tag for elem in format_elements),
            "has_markdown": any(elem.tag.startswith("#") or elem.tag.startswith("*") for elem in format_elements)
        }
        
        prompt = f"""
        Reassemble this luxury fashion content with translations while perfectly preserving formatting:
        
        Brand Context:
        - Brand: {brand_context.brand_name}
        - Voice: {brand_context.brand_voice}
        - Target Market: {brand_context.target_market}
        
        Original Content:
        {original_content}
        
        Translation Mapping:
        {json.dumps(translation_mapping, indent=2, ensure_ascii=False)}
        
        Format Elements Found: {format_context["total_elements"]} elements
        - Element Types: {', '.join(format_context["element_types"])}
        - Contains HTML: {format_context["has_html"]}
        - Contains Markdown: {format_context["has_markdown"]}
        
        Requirements:
        1. Replace original text with translations EXACTLY as provided in the mapping
        2. Preserve ALL HTML tags, attributes, and structure perfectly
        3. Maintain ALL Markdown syntax (headers, links, formatting) exactly
        4. Keep original whitespace, indentation, and line breaks
        5. Preserve any special characters, symbols, or encoded entities
        6. Ensure the translated text flows naturally within the preserved structure
        7. Do not add any explanatory text or comments
        
        Return ONLY the reassembled content with preserved formatting and translated text.
        """
        
        # Use the content reassembly agent
        reassembly_agent = get_content_reassembly_agent()
        logger.info("Executing content reassembly with specialized agent")
        
        run_result = await reassembly_agent.run(prompt, deps=brand_context)
        reassembled_content = run_result.output
        
        logger.info(f"Content reassembly completed. Original length: {len(original_content)}, "
                   f"Reassembled length: {len(reassembled_content)}")
        
        return reassembled_content

    def get_workflow_status(self, task_id: str) -> Optional[WorkflowState]:
        """Get current workflow status"""
        return self.active_workflows.get(task_id)

    async def cancel_workflow(self, task_id: str) -> bool:
        """Cancel a running workflow"""
        if task_id in self.active_workflows:
            workflow = self.active_workflows[task_id]
            workflow.status = "cancelled"
            workflow.updated_at = datetime.utcnow()
            return True
        return False
