# Luxury Translation AI Engine

A specialized AI-powered translation service designed for luxury fashion retailers, delivering brand-consistent, ready-to-publish translations with perfect formatting preservation.

## Overview

This MVP addresses the critical gap between expensive translation agencies and generic AI translation tools. Our solution specifically targets luxury fashion retailers who require premium translation quality that maintains brand voice and cultural sensitivity while preserving complex formatting structures.

The system leverages multiple specialized AI agents orchestrated through Pydantic AI workflows to deliver translations that sound natural and maintain the aspirational quality expected in luxury markets.

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (required)
- Anthropic API key (optional, falls back to OpenAI)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd luxury-translation-ai
```

2. Install dependencies using uv (recommended):
```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync
```

3. Configure environment variables:
```bash
# Copy and edit the environment file
cp .env.example .env

# Add your API keys to .env file
OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here  # optional
```

### Running the Service

Start the FastAPI server:
```bash
uv run uvicorn src.app:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Testing

Run the test suite:
```bash
# Unit tests
uv run pytest tests/test_segmentation.py::TestSegmentationService -v

# API integration tests  
uv run pytest tests/test_segmentation.py::TestSegmentationAPI -v

# End-to-end workflow tests
uv run pytest tests/test_e2e.py -v
```

## Major Design Decisions

### Multi-Agent Architecture

We implemented a specialized agent orchestration system using Pydantic AI rather than a single monolithic translation model. This decision was driven by the complexity of requirements:

- **Format Analysis Agent**: Intelligently segments content while preserving HTML/Markdown structures
- **Luxury Translation Agent**: Specialized in fashion terminology and brand voice preservation  
- **Quality Assessment Agent**: Multi-dimensional evaluation ensuring translations meet luxury standards
- **Content Reassembly Agent**: Perfect reconstruction of formatted content post-translation

This approach allows each agent to focus on its expertise while maintaining type safety and structured outputs.

### Brand Context Integration

Unlike generic translation services, our system requires explicit brand context for every translation. This includes:

- Brand voice characteristics
- Target market cultural considerations
- Custom terminology glossaries
- Quality thresholds specific to luxury positioning

This ensures translations maintain brand consistency and cultural appropriateness across languages.

### Quality Gate System

We implemented a rigorous quality assessment framework with five dimensions:

1. Semantic accuracy
2. Language fluency
3. Brand voice consistency
4. Cultural appropriateness
5. Luxury market positioning

Translations failing quality gates are flagged for human review, ensuring only ready-to-publish content reaches clients.

### Format Preservation Strategy

Rather than treating formatting as an afterthought, our system analyzes and preserves structure throughout the translation process. The Format Analysis Agent extracts translatable segments while maintaining context about their formatting requirements, enabling perfect reconstruction.

## Implementation Findings

### Technical Feasibility

The MVP demonstrates strong technical feasibility. Our multi-agent approach successfully handles:

- Complex HTML and Markdown content with structure preservation
- Brand-aware translation that maintains luxury positioning
- Quality assessment with measurable metrics
- Real-time processing with reasonable latency

### Key Strengths

1. **Specialized Domain Knowledge**: The system understands luxury fashion terminology and cultural nuances
2. **Format Integrity**: Perfect preservation of complex formatting structures
3. **Quality Assurance**: Built-in quality gates prevent substandard translations from reaching production
4. **Brand Consistency**: Explicit brand context ensures translations align with client voice and values
5. **Scalability**: Agent-based architecture allows independent scaling of different capabilities

### Potential Complications

#### Technical Challenges

1. **API Dependencies**: Heavy reliance on external LLM providers (OpenAI/Anthropic) creates potential points of failure and cost considerations
2. **Processing Latency**: Complex multi-agent workflows may be slower than simple translation APIs, though quality justifies the trade-off
3. **Context Limits**: Very large documents may exceed agent context windows, requiring intelligent chunking strategies
4. **Cost Management**: Multiple agent calls per translation could result in higher operational costs than expected

#### Business Challenges

1. **Quality Consistency**: While quality gates help, maintaining consistent luxury standards across different languages and cultural contexts remains challenging
2. **Customization Complexity**: Each luxury brand has unique voice requirements that may require extensive customization
3. **Cultural Expertise**: Certain market-specific translations may still require human cultural experts for final validation
4. **Integration Overhead**: Clients will need to provide detailed brand context and terminology, which may require significant onboarding effort

## Product Discovery Questions

Based on our findings, we recommend exploring these questions with potential clients:

### Content and Volume Requirements

- What types of content do you translate most frequently? (product descriptions, marketing materials, technical documentation)
- What is your typical content volume per month, and how does this vary seasonally?
- Do you work with content in multiple formats (HTML, Markdown, plain text, PDF)?

### Brand and Quality Standards

- How do you currently maintain brand voice consistency across languages and markets?
- Do you have existing brand glossaries or terminology guides for different markets?
- What quality control processes do you use today, and where do they fall short?

### Workflow and Integration

- How does translation fit into your current content workflow?
- Who reviews and approves translations before publication, human in the loop or fully automated?

### Market and Competition

- Which markets are most important for your translation needs?
- How do you currently choose between translation agencies and AI tools?
- What would make you switch from your current translation solution?
- How do you measure ROI on translation investments?

### Technical and Operational

- Do you have specific formatting requirements or constraints?
- How important is translation speed versus quality for different content types?

## API Usage Examples

### Basic Content Segmentation
```bash
curl -X POST "http://localhost:8000/content/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "<h1>Luxury Collection</h1><p>Our exclusive handbags...</p>",
    "segmentation_method": "basic"
  }'
```

### AI-Powered Translation Workflow
```bash
curl -X POST "http://localhost:8000/translate/pydantic-ai" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "<h1>Venetian Artisan Collection</h1><p>Experience luxury craftsmanship...</p>",
    "source_language": "en",
    "target_language": "fr", 
    "brand_name": "Venetian Artisan",
    "brand_voice": "Sophisticated Italian luxury emphasizing heritage",
    "target_market": "French luxury consumers",
    "brand_glossary": {
      "craftsmanship": "savoir-faire",
      "heritage": "patrimoine"
    }
  }'
```

The system is ready for production deployment and can be extended with additional features based on client feedback and requirements.