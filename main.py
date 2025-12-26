
# from fastapi import FastAPI, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import google.generativeai as genai
# import os
# from datetime import datetime
# import logging
# import re
# from typing import Dict, Any, List
# import json

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="HealthGuard AI - Misinformation Detection API",
#     description="AI-powered system to detect health misinformation in online content",
#     version="2.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Setup Gemini API
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Health topics for focused detection
# HEALTH_TOPICS = [
#     "vaccine", "cancer", "covid", "nutrition", "mental_health",
#     "treatment", "prevention", "medication", "diagnosis", "surgery",
#     "alternative_medicine", "supplements", "diet", "exercise", "wellness"
# ]

# # Risk levels with descriptions
# RISK_LEVELS = {
#     "low": {"min": 0, "max": 30, "color": "green", "description": "Likely reliable information"},
#     "medium": {"min": 31, "max": 70, "color": "yellow", "description": "Potentially misleading"},
#     "high": {"min": 71, "max": 90, "color": "orange", "description": "Probably misinformation"},
#     "critical": {"min": 91, "max": 100, "color": "red", "description": "Dangerous misinformation"}
# }

# def setup_gemini():
#     """Initialize Gemini AI with fallback models"""
#     if not GEMINI_API_KEY:
#         logger.warning("No GEMINI_API_KEY found in environment variables")
#         return None, "No API key found"

#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         logger.info("Gemini API configured successfully")
#     except Exception as e:
#         logger.error(f"Gemini configuration failed: {e}")
#         return None, f"API configuration failed: {e}"

#     # Try available models
#     model_priority = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
#     for model_name in model_priority:
#         try:
#             model = genai.GenerativeModel(model_name)
#             # Test with simple prompt
#             response = model.generate_content("Hello")
#             if getattr(response, "text", None):
#                 logger.info(f"Successfully connected to {model_name}")
#                 return model, f"Connected to {model_name}"
#         except Exception as e:
#             logger.warning(f"Failed to connect to {model_name}: {e}")
#             continue

#     logger.error("No compatible Gemini model found")
#     return None, "No compatible Gemini model found"

# gemini_model, setup_status = setup_gemini()
# GEMINI_AVAILABLE = gemini_model is not None
# logger.info(f"Gemini available: {GEMINI_AVAILABLE}")

# def detect_health_topic(text: str) -> List[str]:
#     """Detect which health topics are mentioned in the text"""
#     detected_topics = []
#     text_lower = text.lower()
    
#     topic_keywords = {
#         "vaccine": ["vaccine", "vaccination", "immunization", "inoculation", "shot", "jab", "vax"],
#         "cancer": ["cancer", "tumor", "chemotherapy", "oncology", "malignant", "benign"],
#         "covid": ["covid", "coronavirus", "pandemic", "sars-cov-2", "corona"],
#         "nutrition": ["nutrition", "diet", "food", "eat", "meal", "calorie", "vitamin", "mineral"],
#         "mental_health": ["mental", "depression", "anxiety", "therapy", "psychology", "stress"],
#         "treatment": ["treatment", "therapy", "cure", "medication", "drug", "prescription"],
#         "prevention": ["prevent", "prevention", "avoid", "protection", "safety"],
#         "medication": ["medicine", "drug", "pill", "tablet", "prescription", "pharmacy"],
#         "diagnosis": ["diagnosis", "diagnose", "test", "symptom", "screening"],
#         "surgery": ["surgery", "operation", "surgeon", "operative", "incision"]
#     }
    
#     for topic, keywords in topic_keywords.items():
#         if any(keyword in text_lower for keyword in keywords):
#             detected_topics.append(topic)
    
#     return detected_topics if detected_topics else ["general_health"]

# def analyze_health_text_with_gemini(text: str, detected_topics: List[str]) -> Dict[str, Any]:
#     """
#     Analyze health text for misinformation using Gemini AI
#     """
#     try:
#         topics_str = ", ".join(detected_topics)
        
#         prompt = f"""
#         You are an expert medical fact-checker and health misinformation analyst.
        
#         Analyze this health-related text and determine if it contains misinformation.
        
#         TEXT TO ANALYZE:
#         "{text}"
        
#         Detected health topics: {topics_str}
        
#         Provide your analysis in this EXACT JSON format:
#         {{
#             "is_misinformation": boolean (true if contains misinformation, false if reliable),
#             "confidence_score": number between 0-100 (how confident you are in your assessment),
#             "risk_level": string ("low", "medium", "high", or "critical"),
#             "verdict": string (one sentence summary),
#             "explanation": string (detailed explanation of why it's misinformation/reliable),
#             "key_issues": array of strings (list specific misinformation patterns found),
#             "evidence_needed": array of strings (what evidence would verify/disprove claims),
#             "recommended_action": string (what the user should do with this information)
#         }}
        
#         ANALYSIS GUIDELINES:
#         1. MISINFORMATION INDICATORS:
#            - Makes unsubstantiated cure/treatment claims
#            - Uses fear-mongering or panic-inducing language
#            - References unverified "studies" or "doctors"
#            - Contains conspiracy theories about medical establishments
#            - Makes absolute claims ("100% effective", "never fails")
#            - Contradicts established medical consensus (CDC, WHO, medical journals)
#            - Promotes dangerous medical advice
#            - Uses anecdotal evidence as proof
        
#         2. RELIABLE INDICATORS:
#            - Cites credible sources (CDC, WHO, medical journals, reputable hospitals)
#            - Presents balanced information
#            - Acknowledges limitations/uncertainties
#            - Uses measured, scientific language
#            - Recommends consulting healthcare professionals
        
#         3. SPECIAL CONSIDERATIONS:
#            - Distinguish between "alternative medicine" (may be personal choice) vs "dangerous misinformation"
#            - Consider if the information could cause direct harm
#            - Be cautious with emerging research vs established facts
        
#         Provide ONLY the JSON object, no additional text.
#         """
        
#         response = gemini_model.generate_content(prompt)
#         response_text = getattr(response, "text", str(response))
        
#         # Clean the response to extract JSON
#         response_text = response_text.strip()
#         if response_text.startswith("```json"):
#             response_text = response_text[7:-3].strip()
#         elif response_text.startswith("```"):
#             response_text = response_text[3:-3].strip()
        
#         try:
#             analysis = json.loads(response_text)
            
#             # Validate required fields
#             required_fields = ["is_misinformation", "confidence_score", "risk_level", 
#                              "verdict", "explanation", "key_issues", "evidence_needed", 
#                              "recommended_action"]
            
#             for field in required_fields:
#                 if field not in analysis:
#                     raise ValueError(f"Missing field: {field}")
            
#             # Ensure confidence_score is within range
#             analysis["confidence_score"] = max(0, min(100, int(analysis["confidence_score"])))
            
#             # Ensure risk_level is valid
#             if analysis["risk_level"] not in RISK_LEVELS:
#                 analysis["risk_level"] = "medium"
            
#             return analysis
            
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to parse Gemini response as JSON: {e}")
#             raise Exception("AI response parsing failed")
            
#     except Exception as e:
#         logger.error(f"Gemini analysis failed: {e}")
#         raise Exception(f"AI analysis failed: {str(e)}")

# def get_demo_analysis(text: str, detected_topics: List[str]) -> Dict[str, Any]:
#     """
#     Fallback demo analysis when Gemini is not available
#     """
#     topics_str = ", ".join(detected_topics)
    
#     # Simple rule-based detection for demo
#     text_lower = text.lower()
    
#     # Misinformation indicators
#     misinfo_keywords = [
#         "miracle cure", "100% effective", "big pharma hiding", "secret doctors hate",
#         "government cover-up", "they don't want you to know", "overnight cure",
#         "instantly cures", "medical establishment suppressing", "natural cure for cancer",
#         "vaccines cause autism", "covid hoax", "mask causes oxygen deficiency",
#         "bleach cure", "ivermectin miracle", "hydroxychloroquine cure"
#     ]
    
#     # Reliable indicators
#     reliable_keywords = [
#         "according to cdc", "who recommends", "clinical study shows", "peer-reviewed research",
#         "consult your doctor", "evidence suggests", "medical consensus", "scientific evidence",
#         "may help with", "can reduce risk", "studies indicate", "health authorities recommend"
#     ]
    
#     # Count indicators
#     misinfo_count = sum(1 for keyword in misinfo_keywords if keyword in text_lower)
#     reliable_count = sum(1 for keyword in reliable_keywords if keyword in text_lower)
    
#     # Determine if misinformation
#     if misinfo_count > reliable_count:
#         is_misinfo = True
#         confidence = min(80 + (misinfo_count * 5), 95)
#         risk_level = "high" if misinfo_count > 2 else "medium"
#         verdict = "Likely contains health misinformation"
#         explanation = f"Text contains {misinfo_count} common misinformation patterns including claims of miraculous cures, conspiracy theories, or unsubstantiated medical advice."
#     else:
#         is_misinfo = False
#         confidence = min(70 + (reliable_count * 5), 90)
#         risk_level = "low" if reliable_count > 0 else "medium"
#         verdict = "Appears to be reliable health information"
#         explanation = f"Text contains {reliable_count} indicators of reliable health information such as references to established sources or measured language."
    
#     return {
#         "is_misinformation": is_misinfo,
#         "confidence_score": confidence,
#         "risk_level": risk_level,
#         "verdict": verdict,
#         "explanation": explanation,
#         "key_issues": [
#             "Analyzed based on keyword patterns",
#             f"Detected topics: {topics_str}",
#             f"Misinformation indicators found: {misinfo_count}",
#             f"Reliability indicators found: {reliable_count}"
#         ],
#         "evidence_needed": [
#             "Scientific studies from reputable journals",
#             "Guidelines from health authorities (CDC, WHO)",
#             "Expert medical opinions",
#             "Clinical trial evidence"
#         ],
#         "recommended_action": "Verify claims with healthcare professionals and official health authorities. Do not make medical decisions based on unverified online information."
#     }

# def calculate_risk_metrics(text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
#     """Calculate additional risk metrics based on text analysis"""
#     text_lower = text.lower()
    
#     # Calculate urgency score
#     urgency_words = ["urgent", "emergency", "warning", "alert", "breaking", "immediately"]
#     urgency_score = sum(1 for word in urgency_words if word in text_lower)
    
#     # Calculate sensationalism score
#     sensational_words = ["shocking", "amazing", "miracle", "secret", "hidden", "unbelievable"]
#     sensational_score = sum(1 for word in sensational_words if word in text_lower)
    
#     # Calculate conspiracy score
#     conspiracy_words = ["they", "them", "government", "big pharma", "cover-up", "suppress"]
#     conspiracy_score = sum(1 for word in conspiracy_words if word in text_lower)
    
#     # Calculate authority appeal score
#     authority_words = ["doctor says", "expert reveals", "study proves", "research shows"]
#     authority_score = sum(1 for word in authority_words if word in text_lower)
    
#     # Text statistics
#     char_count = len(text)
#     word_count = len(text.split())
#     exclamation_count = text.count('!')
#     question_count = text.count('?')
#     all_caps_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
    
#     return {
#         "text_metrics": {
#             "character_count": char_count,
#             "word_count": word_count,
#             "exclamation_count": exclamation_count,
#             "question_count": question_count,
#             "all_caps_ratio": round(all_caps_ratio, 3)
#         },
#         "pattern_scores": {
#             "urgency": urgency_score,
#             "sensationalism": sensational_score,
#             "conspiracy_language": conspiracy_score,
#             "authority_appeal": authority_score
#         },
#         "overall_risk_score": analysis["confidence_score"] if analysis["is_misinformation"] else 100 - analysis["confidence_score"]
#     }

# @app.get("/")
# async def root():
#     return {
#         "message": "HealthGuard AI - Health Misinformation Detection API",
#         "status": "active",
#         "service": "AI-powered health misinformation detection",
#         "version": "2.0.0",
#         "gemini_available": GEMINI_AVAILABLE,
#         "endpoints": {
#             "POST /analyze": "Analyze health text for misinformation",
#             "GET /health": "Service health check",
#             "GET /topics": "List supported health topics"
#         }
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "gemini_available": GEMINI_AVAILABLE,
#         "setup_status": setup_status,
#         "supported_health_topics": HEALTH_TOPICS,
#         "risk_levels": RISK_LEVELS
#     }

# @app.get("/topics")
# async def get_supported_topics():
#     """Get list of health topics the system can detect"""
#     return {
#         "health_topics": HEALTH_TOPICS,
#         "count": len(HEALTH_TOPICS),
#         "description": "These health topics are automatically detected in the text"
#     }

# @app.post("/analyze")
# async def analyze_health_text(
#     text: str = Form(..., description="Health-related text to analyze"),
#     use_ai: bool = Form(True, description="Use AI analysis (if False, uses rule-based demo)"),
#     detailed_metrics: bool = Form(True, description="Include detailed text metrics in response")
# ):
#     """
#     Analyze health-related text for misinformation
    
#     Parameters:
#     - text: The health-related text to analyze (required)
#     - use_ai: Whether to use AI analysis (default: True)
#     - detailed_metrics: Include detailed text analysis metrics (default: True)
    
#     Returns:
#     - Analysis of whether the text contains misinformation
#     - Risk level and confidence score
#     - Detailed explanation and recommendations
#     """
#     try:
#         logger.info(f"Health analysis request received: {len(text)} characters")
        
#         # Validate input
#         if not text or len(text.strip()) < 10:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Text must be at least 10 characters long"
#             )
        
#         if len(text) > 10000:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Text must be less than 10,000 characters"
#             )
        
#         # Detect health topics
#         detected_topics = detect_health_topic(text)
#         logger.info(f"Detected health topics: {detected_topics}")
        
#         # Perform analysis
#         if use_ai and GEMINI_AVAILABLE:
#             analysis = analyze_health_text_with_gemini(text, detected_topics)
#             analysis_source = "Gemini AI"
#         else:
#             analysis = get_demo_analysis(text, detected_topics)
#             analysis_source = "Rule-based analysis (Demo Mode)"
        
#         # Calculate additional metrics if requested
#         additional_metrics = {}
#         if detailed_metrics:
#             additional_metrics = calculate_risk_metrics(text, analysis)
        
#         # Prepare response
#         response = {
#             "success": True,
#             "analysis": {
#                 "text_preview": text[:200] + "..." if len(text) > 200 else text,
#                 "text_length": len(text),
#                 "detected_topics": detected_topics,
#                 "is_misinformation": analysis["is_misinformation"],
#                 "verdict": analysis["verdict"],
#                 "confidence_score": analysis["confidence_score"],
#                 "risk_level": analysis["risk_level"],
#                 "risk_description": RISK_LEVELS.get(analysis["risk_level"], {}).get("description", "Unknown risk"),
#                 "explanation": analysis["explanation"],
#                 "key_issues": analysis["key_issues"],
#                 "evidence_needed": analysis["evidence_needed"],
#                 "recommended_action": analysis["recommended_action"],
#                 "analysis_source": analysis_source,
#                 "timestamp": datetime.now().isoformat()
#             }
#         }
        
#         # Add detailed metrics if requested
#         if additional_metrics:
#             response["detailed_metrics"] = additional_metrics
        
#         logger.info(f"Analysis completed: {analysis['verdict']} (confidence: {analysis['confidence_score']}%)")
        
#         return response
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Health analysis error: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Analysis failed: {str(e)}"
#         )

# @app.post("/batch_analyze")
# async def batch_analyze_texts(
#     texts: List[str] = Form(..., description="List of health texts to analyze"),
#     use_ai: bool = Form(True, description="Use AI analysis")
# ):
#     """
#     Analyze multiple health texts in batch
    
#     Parameters:
#     - texts: List of health-related texts to analyze
#     - use_ai: Whether to use AI analysis
    
#     Returns:
#     - List of analysis results for each text
#     - Summary statistics
#     """
#     try:
#         if not texts or len(texts) == 0:
#             raise HTTPException(
#                 status_code=400,
#                 detail="At least one text is required"
#             )
        
#         if len(texts) > 100:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Maximum 100 texts per batch"
#             )
        
#         results = []
#         misinfo_count = 0
#         high_risk_count = 0
        
#         for i, text in enumerate(texts):
#             try:
#                 # Detect topics
#                 detected_topics = detect_health_topic(text)
                
#                 # Perform analysis
#                 if use_ai and GEMINI_AVAILABLE:
#                     analysis = analyze_health_text_with_gemini(text, detected_topics)
#                     source = "Gemini AI"
#                 else:
#                     analysis = get_demo_analysis(text, detected_topics)
#                     source = "Rule-based analysis"
                
#                 # Count statistics
#                 if analysis["is_misinformation"]:
#                     misinfo_count += 1
                
#                 if analysis["risk_level"] in ["high", "critical"]:
#                     high_risk_count += 1
                
#                 results.append({
#                     "text_id": i + 1,
#                     "text_preview": text[:100] + "..." if len(text) > 100 else text,
#                     "detected_topics": detected_topics,
#                     "is_misinformation": analysis["is_misinformation"],
#                     "confidence_score": analysis["confidence_score"],
#                     "risk_level": analysis["risk_level"],
#                     "verdict": analysis["verdict"],
#                     "analysis_source": source
#                 })
                
#             except Exception as e:
#                 results.append({
#                     "text_id": i + 1,
#                     "text_preview": text[:100] + "..." if len(text) > 100 else text,
#                     "error": str(e),
#                     "analysis_source": "Failed"
#                 })
        
#         # Calculate summary
#         total_texts = len(texts)
#         successful_analyses = len([r for r in results if "error" not in r])
        
#         return {
#             "success": True,
#             "summary": {
#                 "total_texts": total_texts,
#                 "successful_analyses": successful_analyses,
#                 "misinformation_found": misinfo_count,
#                 "high_risk_texts": high_risk_count,
#                 "misinformation_rate": round((misinfo_count / successful_analyses) * 100, 2) if successful_analyses > 0 else 0,
#                 "timestamp": datetime.now().isoformat()
#             },
#             "results": results
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Batch analysis error: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Batch analysis failed: {str(e)}"
#         )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, Form, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import logging
import re
from typing import Dict, Any, List, Optional
import json
import uuid
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedVerax AI - Health Misinformation Detection API",
    description="AI-powered system for detecting and analyzing health misinformation",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    text: str
    source: Optional[str] = "unknown"
    audience_reach: Optional[int] = 50
    publication_date: Optional[str] = None
    use_ai: Optional[bool] = True

class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    use_ai: Optional[bool] = True

class SimulationRequest(BaseModel):
    text_length: int = 500
    has_credible_source: bool = False
    has_citations: bool = False
    recently_published: bool = True
    sensational_language: int = 50
    medical_terms: int = 30
    author_credibility: int = 50
    emotional_language: bool = False

# Setup Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Update health topics to match frontend categories
HEALTH_TOPICS = [
    "vaccine", "cancer", "covid", "nutrition", "mental_health",
    "alternative_medicine", "supplements", "diet", "exercise", "wellness",
    "treatment", "prevention", "medication", "diagnosis", "surgery"
]

# Frontend-compatible risk levels with proper color mapping
RISK_LEVELS = {
    "low": {
        "min": 0, "max": 30, 
        "color": "hsl(152 69% 40%)",  # Frontend's risk-safe
        "description": "Likely reliable information",
        "classification": "Reliable"
    },
    "medium": {
        "min": 31, "max": 60, 
        "color": "hsl(43 96% 50%)",   # Frontend's risk-moderate
        "description": "Potentially misleading",
        "classification": "Moderate Risk"
    },
    "high": {
        "min": 61, "max": 80, 
        "color": "hsl(25 95% 53%)",   # Frontend's risk-high
        "description": "Probably misinformation",
        "classification": "High Risk"
    },
    "critical": {
        "min": 81, "max": 100, 
        "color": "hsl(0 72% 51%)",    # Frontend's risk-critical
        "description": "Dangerous misinformation",
        "classification": "Critical"
    }
}

# In-memory storage for history (in production, use a database)
analysis_history = []

def setup_gemini():
    """Initialize Gemini AI with fallback models"""
    if not GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY found in environment variables")
        return None, "No API key found"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Gemini configuration failed: {e}")
        return None, f"API configuration failed: {e}"

    # Try available models
    model_priority = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
    for model_name in model_priority:
        try:
            model = genai.GenerativeModel(model_name)
            # Test with simple prompt
            response = model.generate_content("Hello")
            if getattr(response, "text", None):
                logger.info(f"Successfully connected to {model_name}")
                return model, f"Connected to {model_name}"
        except Exception as e:
            logger.warning(f"Failed to connect to {model_name}: {e}")
            continue

    logger.error("No compatible Gemini model found")
    return None, "No compatible Gemini model found"

gemini_model, setup_status = setup_gemini()
GEMINI_AVAILABLE = gemini_model is not None
logger.info(f"Gemini available: {GEMINI_AVAILABLE}")

def detect_health_topic(text: str) -> List[str]:
    """Detect which health topics are mentioned in the text"""
    detected_topics = []
    text_lower = text.lower()
    
    topic_keywords = {
        "vaccine": ["vaccine", "vaccination", "immunization", "inoculation", "shot", "jab", "vax"],
        "cancer": ["cancer", "tumor", "chemotherapy", "oncology", "malignant", "benign"],
        "covid": ["covid", "coronavirus", "pandemic", "sars-cov-2", "corona", "lockdown"],
        "nutrition": ["nutrition", "diet", "food", "eat", "meal", "calorie", "vitamin", "mineral"],
        "mental_health": ["mental", "depression", "anxiety", "therapy", "psychology", "stress", "psychiatric"],
        "alternative_medicine": ["alternative", "herbal", "holistic", "natural remedy", "acupuncture", "chiropractic"],
        "treatment": ["treatment", "therapy", "cure", "medication", "drug", "prescription", "remedy"],
        "prevention": ["prevent", "prevention", "avoid", "protection", "safety", "vaccinate"],
        "medication": ["medicine", "drug", "pill", "tablet", "prescription", "pharmacy", "antibiotic"],
        "diagnosis": ["diagnosis", "diagnose", "test", "symptom", "screening", "biopsy", "scan"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_topics.append(topic)
    
    return detected_topics if detected_topics else ["general_health"]

def analyze_health_text_with_gemini(text: str, detected_topics: List[str], source: str = "unknown") -> Dict[str, Any]:
    """
    Analyze health text for misinformation using Gemini AI
    Returns frontend-compatible analysis structure
    """
    try:
        topics_str = ", ".join(detected_topics)
        
        # Enhanced prompt for frontend-compatible response
        prompt = f"""
        You are MedVerax, an expert medical fact-checking AI. Analyze this health content and provide a detailed misinformation assessment.
        
        TEXT TO ANALYZE: "{text}"
        
        SOURCE CONTEXT: {source}
        DETECTED TOPICS: {topics_str}
        
        Provide your analysis in this EXACT JSON format:
        {{
            "is_misinformation": boolean,
            "confidence_score": number (0-100),
            "risk_score": number (0-100),
            "risk_level": string ("low", "medium", "high", "critical"),
            "classification": string ("Reliable", "Low Risk", "Moderate Risk", "High Risk", "Critical"),
            "verdict": string (concise summary),
            "explanation": string (detailed reasoning, 2-3 sentences),
            "language_cues": array of strings (specific problematic language patterns),
            "claim_indicators": array of strings (types of claims made),
            "missing_references": array of strings (what evidence is lacking),
            "highlighted_phrases": array of objects [{{"text": string, "type": "danger"/"warning"/"info"}}],
            "evidence_needed": array of strings,
            "recommended_action": string
        }}
        
        ANALYSIS CRITERIA:
        1. Check for: Absolute claims, conspiracy theories, fear-mongering, unverified sources
        2. Look for: Scientific references, balanced language, expert citations
        3. Consider: Source credibility, audience impact, potential harm
        
        RISK CLASSIFICATION GUIDE:
        - 0-30: Reliable (green) - Well-sourced, balanced, scientific
        - 31-60: Moderate Risk (yellow) - Some concerns, needs verification
        - 61-80: High Risk (orange) - Likely misinformation, problematic
        - 81-100: Critical (red) - Dangerous misinformation, immediate concern
        
        Return ONLY the JSON object.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = getattr(response, "text", str(response))
        
        # Clean the response
        response_text = response_text.strip()
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'^```\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        
        analysis = json.loads(response_text)
        
        # Ensure frontend compatibility
        analysis["risk_score"] = analysis.get("risk_score", analysis.get("confidence_score", 50))
        
        # Map risk_score to risk_level if not provided
        if "risk_level" not in analysis:
            risk_score = analysis["risk_score"]
            if risk_score <= 30:
                analysis["risk_level"] = "low"
                analysis["classification"] = "Reliable"
            elif risk_score <= 60:
                analysis["risk_level"] = "medium"
                analysis["classification"] = "Moderate Risk"
            elif risk_score <= 80:
                analysis["risk_level"] = "high"
                analysis["classification"] = "High Risk"
            else:
                analysis["risk_level"] = "critical"
                analysis["classification"] = "Critical"
        
        # Ensure required arrays exist
        analysis["language_cues"] = analysis.get("language_cues", [])
        analysis["claim_indicators"] = analysis.get("claim_indicators", [])
        analysis["missing_references"] = analysis.get("missing_references", [])
        analysis["highlighted_phrases"] = analysis.get("highlighted_phrases", [])
        analysis["evidence_needed"] = analysis.get("evidence_needed", [])
        
        return analysis
        
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        raise Exception(f"AI analysis failed: {str(e)}")

def get_frontend_compatible_analysis(text: str, detected_topics: List[str], source: str = "unknown") -> Dict[str, Any]:
    """
    Generate frontend-compatible analysis structure
    Used when Gemini is not available
    """
    text_lower = text.lower()
    
    # Enhanced keyword detection
    misinfo_phrases = [
        {"text": "miracle cure", "type": "danger"},
        {"text": "100% effective", "type": "danger"},
        {"text": "big pharma hiding", "type": "danger"},
        {"text": "overnight cure", "type": "danger"},
        {"text": "vaccines cause", "type": "danger"},
        {"text": "covid hoax", "type": "danger"},
        {"text": "secret remedy", "type": "warning"},
        {"text": "they don't want you to know", "type": "warning"},
        {"text": "natural cure for cancer", "type": "warning"}
    ]
    
    reliable_phrases = [
        {"text": "according to studies", "type": "info"},
        {"text": "clinical evidence", "type": "info"},
        {"text": "research shows", "type": "info"},
        {"text": "consult your doctor", "type": "info"}
    ]
    
    # Detect phrases
    detected_phrases = []
    for phrase in misinfo_phrases + reliable_phrases:
        if phrase["text"] in text_lower:
            detected_phrases.append(phrase)
    
    # Calculate risk based on content
    word_count = len(text.split())
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Rule-based scoring
    base_score = 50
    
    # Adjust based on content characteristics
    if any(p["type"] == "danger" for p in detected_phrases):
        base_score += 30
    if any(p["type"] == "warning" for p in detected_phrases):
        base_score += 15
    if any(p["type"] == "info" for p in detected_phrases):
        base_score -= 10
    
    # Text length factor
    if word_count < 100:
        base_score += 10  # Short content often lacks nuance
    elif word_count > 500:
        base_score -= 5   # Longer content often more detailed
    
    # Emotional language detection
    emotional_words = ["urgent", "emergency", "warning", "dangerous", "fear"]
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    base_score += emotional_count * 3
    
    # Cap the score
    risk_score = max(0, min(100, base_score))
    
    # Determine classification
    if risk_score <= 30:
        risk_level = "low"
        classification = "Reliable"
        is_misinfo = False
        verdict = "Content appears reliable based on initial analysis"
    elif risk_score <= 60:
        risk_level = "medium"
        classification = "Moderate Risk"
        is_misinfo = False
        verdict = "Content shows some concerning patterns"
    elif risk_score <= 80:
        risk_level = "high"
        classification = "High Risk"
        is_misinfo = True
        verdict = "Content likely contains misinformation"
    else:
        risk_level = "critical"
        classification = "Critical"
        is_misinfo = True
        verdict = "Content contains dangerous misinformation"
    
    # Generate explanation
    explanation = f"This content scored {risk_score}/100 on our misinformation risk scale. "
    if detected_phrases:
        explanation += f"Detected {len([p for p in detected_phrases if p['type'] in ['danger', 'warning']])} concerning phrases. "
    explanation += "Always verify health information with qualified professionals."
    
    return {
        "is_misinformation": is_misinfo,
        "confidence_score": 85,  # Confidence in our assessment
        "risk_score": risk_score,
        "risk_level": risk_level,
        "classification": classification,
        "verdict": verdict,
        "explanation": explanation,
        "language_cues": [
            "Emotional language detected" if emotional_count > 0 else "Neutral tone",
            f"{exclamation_count} exclamation marks" if exclamation_count > 3 else "Moderate punctuation",
            f"{word_count} words analyzed"
        ],
        "claim_indicators": [
            "Health claims present" if "cure" in text_lower or "treatment" in text_lower else "Informational content",
            "Absolute statements" if "always" in text_lower or "never" in text_lower else "Qualified statements"
        ],
        "missing_references": [
            "Scientific citations needed",
            "Expert sources not referenced"
        ],
        "highlighted_phrases": detected_phrases[:5],  # Limit to 5 phrases
        "evidence_needed": [
            "Peer-reviewed studies",
            "Clinical trial data",
            "Expert medical consensus"
        ],
        "recommended_action": "Consult healthcare professionals and verify with authoritative sources like CDC or WHO before making health decisions."
    }

def calculate_detailed_metrics(text: str, analysis: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    """Calculate detailed metrics for frontend analytics"""
    text_lower = text.lower()
    
    # Pattern detection for analytics
    urgency_words = ["urgent", "emergency", "warning", "alert", "immediately", "now"]
    sensational_words = ["shocking", "amazing", "miracle", "secret", "hidden", "unbelievable"]
    conspiracy_words = ["they", "them", "government", "big pharma", "cover-up", "suppress", "hidden truth"]
    authority_words = ["doctor says", "expert reveals", "study proves", "research shows", "scientists claim"]
    
    # Scores
    urgency_score = sum(1 for word in urgency_words if word in text_lower)
    sensational_score = sum(1 for word in sensational_words if word in text_lower)
    conspiracy_score = sum(1 for word in conspiracy_words if word in text_lower)
    authority_score = sum(1 for word in authority_words if word in text_lower)
    
    # Text statistics
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text))
    avg_word_length = char_count / max(word_count, 1)
    
    # Readability (simplified)
    readability = "Easy" if avg_word_length < 5 else "Moderate" if avg_word_length < 7 else "Complex"
    
    # Topic distribution (for analytics dashboard)
    topics = detect_health_topic(text)
    topic_distribution = {topic: 1 for topic in topics}
    
    return {
        "text_metrics": {
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "readability_level": readability,
            "avg_word_length": round(avg_word_length, 1)
        },
        "pattern_scores": {
            "urgency": min(urgency_score * 20, 100),
            "sensationalism": min(sensational_score * 25, 100),
            "conspiracy_language": min(conspiracy_score * 20, 100),
            "authority_appeal": min(authority_score * 25, 100),
            "emotional_intensity": min((urgency_score + sensational_score) * 15, 100)
        },
        "content_analysis": {
            "has_medical_terms": any(term in text_lower for term in ["treatment", "therapy", "medication", "dose", "symptom"]),
            "has_statistics": bool(re.search(r'\d+%|\d+ out of \d+', text)),
            "has_references": any(ref in text_lower for ref in ["study", "research", "according to", "source:"]),
            "has_calls_to_action": any(cta in text_lower for cta in ["share", "tell everyone", "spread", "forward"])
        },
        "topic_distribution": topic_distribution,
        "source_risk": {
            "source_type": source,
            "source_risk_score": 40 if "social" in source else 20 if "news" in source else 60,
            "trust_factor": 0.8 if "medical" in source else 0.5 if "official" in source else 0.3
        }
    }

def run_simulation(params: SimulationRequest) -> Dict[str, Any]:
    """Run what-if analysis simulation (for simulation.tsx)"""
    risk_score = 50  # Base
    
    # Calculate based on parameters
    if params.text_length < 100:
        risk_score += 15
    elif params.text_length > 1000:
        risk_score -= 10
    
    if params.has_credible_source:
        risk_score -= 20
    
    if params.has_citations:
        risk_score -= 15
    
    risk_score += (params.sensational_language - 50) * 0.3
    risk_score -= (params.medical_terms - 30) * 0.2
    risk_score -= (params.author_credibility - 50) * 0.25
    
    if params.emotional_language:
        risk_score += 12
    
    # Normalize
    risk_score = max(0, min(100, risk_score))
    
    # Determine classification
    if risk_score <= 30:
        risk_level = "low"
        classification = "Reliable"
    elif risk_score <= 60:
        risk_level = "medium"
        classification = "Moderate Risk"
    elif risk_score <= 80:
        risk_level = "high"
        classification = "High Risk"
    else:
        risk_level = "critical"
        classification = "Critical"
    
    # Calculate factor impacts
    factors = []
    factors.append({"factor": "Content Length", "impact": 15 if params.text_length < 100 else -10 if params.text_length > 1000 else 0})
    factors.append({"factor": "Credible Source", "impact": -20 if params.has_credible_source else 10})
    factors.append({"factor": "Citations", "impact": -15 if params.has_citations else 5})
    factors.append({"factor": "Sensational Language", "impact": round((params.sensational_language - 50) * 0.3)})
    factors.append({"factor": "Medical Terms", "impact": round(-(params.medical_terms - 30) * 0.2)})
    factors.append({"factor": "Author Credibility", "impact": round(-(params.author_credibility - 50) * 0.25)})
    factors.append({"factor": "Emotional Language", "impact": 12 if params.emotional_language else 0})
    
    return {
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "classification": classification,
        "factors": factors,
        "parameters": params.dict()
    }

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "MedVerax AI - Health Misinformation Detection API",
        "status": "active",
        "version": "3.0.0",
        "frontend_compatible": True,
        "endpoints": {
            "POST /api/analyze": "Analyze single text (with metadata)",
            "POST /api/batch": "Analyze multiple texts",
            "POST /api/simulate": "Run what-if analysis simulation",
            "GET /api/analytics": "Get analytics dashboard data",
            "GET /api/history": "Get analysis history",
            "GET /api/health": "Health check",
            "GET /api/topics": "Supported health topics"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_available": GEMINI_AVAILABLE,
        "frontend_support": True,
        "version": "3.0.0"
    }

@app.post("/api/analyze")
async def analyze_text(request: AnalysisRequest):
    """
    Main analysis endpoint compatible with detect.tsx frontend
    """
    try:
        logger.info(f"Analysis request: {len(request.text)} chars, source: {request.source}")
        
        # Validate
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(400, "Text must be at least 10 characters")
        if len(request.text) > 10000:
            raise HTTPException(400, "Text too long (max 10,000 chars)")
        
        # Detect topics
        detected_topics = detect_health_topic(request.text)
        
        # Perform analysis
        if request.use_ai and GEMINI_AVAILABLE:
            analysis = analyze_health_text_with_gemini(
                request.text, 
                detected_topics,
                request.source
            )
            analysis_source = "Gemini AI"
        else:
            analysis = get_frontend_compatible_analysis(
                request.text,
                detected_topics,
                request.source
            )
            analysis_source = "Rule-based Analysis"
        
        # Calculate detailed metrics
        detailed_metrics = calculate_detailed_metrics(
            request.text,
            analysis,
            request.source
        )
        
        # Create history entry
        history_entry = {
            "id": str(uuid.uuid4()),
            "text": request.text[:500],  # Store preview
            "full_text_length": len(request.text),
            "source": request.source,
            "audience_reach": request.audience_reach,
            "publication_date": request.publication_date,
            "timestamp": datetime.now().isoformat(),
            "result": analysis,
            "metrics": detailed_metrics
        }
        
        # Store in memory (limit to 1000 entries)
        analysis_history.append(history_entry)
        if len(analysis_history) > 1000:
            analysis_history.pop(0)
        
        # Build frontend-compatible response
        response = {
            "success": True,
            "analysis": {
                # Core results
                "risk_score": analysis["risk_score"],
                "confidence_score": analysis["confidence_score"],
                "risk_level": analysis["risk_level"],
                "classification": analysis["classification"],
                "verdict": analysis["verdict"],
                "explanation": analysis["explanation"],
                "is_misinformation": analysis["is_misinformation"],
                
                # Detailed breakdowns
                "language_cues": analysis["language_cues"],
                "claim_indicators": analysis["claim_indicators"],
                "missing_references": analysis["missing_references"],
                "highlighted_phrases": analysis["highlighted_phrases"],
                "evidence_needed": analysis["evidence_needed"],
                "recommended_action": analysis["recommended_action"],
                
                # Metadata
                "detected_topics": detected_topics,
                "text_preview": request.text[:200] + "..." if len(request.text) > 200 else request.text,
                "text_length": len(request.text),
                "source": request.source,
                "analysis_source": analysis_source,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_metrics": detailed_metrics,
            "history_id": history_entry["id"]
        }
        
        logger.info(f"Analysis complete: {analysis['classification']} ({analysis['risk_score']}%)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.post("/api/batch")
async def batch_analyze(request: BatchAnalysisRequest):
    """Batch analysis for multiple texts"""
    try:
        if not request.texts or len(request.texts) == 0:
            raise HTTPException(400, "At least one text required")
        if len(request.texts) > 50:
            raise HTTPException(400, "Maximum 50 texts per batch")
        
        results = []
        for i, text in enumerate(request.texts):
            try:
                detected_topics = detect_health_topic(text)
                
                if request.use_ai and GEMINI_AVAILABLE:
                    analysis = analyze_health_text_with_gemini(text, detected_topics)
                    source = "Gemini AI"
                else:
                    analysis = get_frontend_compatible_analysis(text, detected_topics)
                    source = "Rule-based"
                
                results.append({
                    "id": i + 1,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "detected_topics": detected_topics,
                    "risk_score": analysis["risk_score"],
                    "risk_level": analysis["risk_level"],
                    "classification": analysis["classification"],
                    "is_misinformation": analysis["is_misinformation"],
                    "verdict": analysis["verdict"],
                    "analysis_source": source
                })
                
            except Exception as e:
                results.append({
                    "id": i + 1,
                    "error": str(e),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                })
        
        # Calculate statistics
        successful = [r for r in results if "error" not in r]
        misinfo_count = sum(1 for r in successful if r.get("is_misinformation", False))
        high_risk_count = sum(1 for r in successful if r.get("risk_level") in ["high", "critical"])
        
        return {
            "success": True,
            "summary": {
                "total": len(request.texts),
                "successful": len(successful),
                "misinformation_count": misinfo_count,
                "high_risk_count": high_risk_count,
                "misinformation_rate": round((misinfo_count / len(successful) * 100), 2) if successful else 0,
                "avg_risk_score": round(sum(r.get("risk_score", 0) for r in successful) / len(successful), 1) if successful else 0
            },
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(500, f"Batch analysis failed: {str(e)}")

@app.post("/api/simulate")
async def simulate_analysis(request: SimulationRequest):
    """What-if analysis simulation endpoint for simulation.tsx"""
    try:
        simulation_result = run_simulation(request)
        
        return {
            "success": True,
            "simulation": simulation_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {str(e)}")

@app.get("/api/analytics")
async def get_analytics(
    time_range: str = "month",
    limit: int = 100
):
    """Get analytics data for dashboard (analytics.tsx)"""
    try:
        # Filter by time range
        cutoff_date = datetime.now()
        if time_range == "week":
            cutoff_date -= timedelta(days=7)
        elif time_range == "month":
            cutoff_date -= timedelta(days=30)
        elif time_range == "quarter":
            cutoff_date -= timedelta(days=90)
        elif time_range == "year":
            cutoff_date -= timedelta(days=365)
        
        # Filter history
        recent_history = [
            h for h in analysis_history[-limit:]
            if datetime.fromisoformat(h["timestamp"].replace('Z', '+00:00')) > cutoff_date
        ]
        
        if not recent_history:
            # Generate demo data for empty history
            return generate_demo_analytics()
        
        # Calculate statistics
        total_analyses = len(recent_history)
        misinfo_count = sum(1 for h in recent_history if h["result"]["is_misinformation"])
        avg_risk = sum(h["result"]["risk_score"] for h in recent_history) / total_analyses
        
        # Topic distribution
        topic_counts = defaultdict(int)
        for h in recent_history:
            for topic in h["result"].get("detected_topics", []):
                topic_counts[topic] += 1
        
        # Risk distribution
        risk_distribution = defaultdict(int)
        for h in recent_history:
            risk_level = h["result"]["risk_level"]
            risk_distribution[risk_level] += 1
        
        # Time series data (simplified)
        time_series = []
        for i in range(12):
            date = f"Month {i+1}"
            misinformation = (i * 15) + 100
            reliable = (i * 20) + 300
            time_series.append({
                "date": date,
                "misinformation": misinformation,
                "reliable": reliable,
                "total": misinformation + reliable
            })
        
        # Source distribution
        sources = defaultdict(int)
        for h in recent_history:
            source = h.get("source", "unknown")
            sources[source] += 1
        
        return {
            "success": True,
            "analytics": {
                "overview": {
                    "total_analyzed": total_analyses,
                    "misinformation_detected": misinfo_count,
                    "reliable_content": total_analyses - misinfo_count,
                    "detection_rate": round((misinfo_count / total_analyses) * 100, 1) if total_analyses > 0 else 0,
                    "avg_risk_score": round(avg_risk, 1)
                },
                "time_series": time_series,
                "topic_distribution": [
                    {"category": topic, "count": count, "color": get_topic_color(topic)}
                    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:6]
                ],
                "risk_distribution": [
                    {"name": "Critical", "value": risk_distribution.get("critical", 0), "color": "hsl(0 72% 51%)"},
                    {"name": "High Risk", "value": risk_distribution.get("high", 0), "color": "hsl(25 95% 53%)"},
                    {"name": "Moderate", "value": risk_distribution.get("medium", 0), "color": "hsl(43 96% 50%)"},
                    {"name": "Low Risk", "value": risk_distribution.get("low", 0), "color": "hsl(152 60% 45%)"},
                    {"name": "Reliable", "value": total_analyses - sum(risk_distribution.values()), "color": "hsl(152 69% 40%)"}
                ],
                "source_analysis": [
                    {"source": source, "percentage": round((count / total_analyses) * 100), "risk": 70 if "social" in source else 40}
                    for source, count in list(sources.items())[:5]
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return generate_demo_analytics()

def generate_demo_analytics():
    """Generate demo analytics data when no history exists"""
    return {
        "success": True,
        "analytics": {
            "overview": {
                "total_analyzed": 12847,
                "misinformation_detected": 2456,
                "reliable_content": 10391,
                "detection_rate": 19.1,
                "avg_risk_score": 42.5
            },
            "time_series": [
                {"date": "Jan", "misinformation": 120, "reliable": 340, "total": 460},
                {"date": "Feb", "misinformation": 145, "reliable": 380, "total": 525},
                {"date": "Mar", "misinformation": 190, "reliable": 420, "total": 610},
                {"date": "Apr", "misinformation": 160, "reliable": 390, "total": 550},
                {"date": "May", "misinformation": 210, "reliable": 450, "total": 660},
                {"date": "Jun", "misinformation": 185, "reliable": 480, "total": 665},
                {"date": "Jul", "misinformation": 230, "reliable": 520, "total": 750},
                {"date": "Aug", "misinformation": 195, "reliable": 490, "total": 685},
                {"date": "Sep", "misinformation": 250, "reliable": 540, "total": 790},
                {"date": "Oct", "misinformation": 280, "reliable": 580, "total": 860},
                {"date": "Nov", "misinformation": 240, "reliable": 560, "total": 800},
                {"date": "Dec", "misinformation": 310, "reliable": 620, "total": 930}
            ],
            "topic_distribution": [
                {"category": "Vaccines", "count": 450, "color": "hsl(0 72% 51%)"},
                {"category": "Alternative Medicine", "count": 320, "color": "hsl(25 95% 53%)"},
                {"category": "Nutrition", "count": 280, "color": "hsl(43 96% 50%)"},
                {"category": "Mental Health", "count": 190, "color": "hsl(152 69% 40%)"},
                {"category": "COVID-19", "count": 380, "color": "hsl(175 55% 32%)"},
                {"category": "Cancer", "count": 220, "color": "hsl(200 70% 50%)"}
            ],
            "risk_distribution": [
                {"name": "Critical", "value": 15, "color": "hsl(0 72% 51%)"},
                {"name": "High Risk", "value": 25, "color": "hsl(25 95% 53%)"},
                {"name": "Moderate", "value": 30, "color": "hsl(43 96% 50%)"},
                {"name": "Low Risk", "value": 18, "color": "hsl(152 60% 45%)"},
                {"name": "Reliable", "value": 12, "color": "hsl(152 69% 40%)"}
            ],
            "source_analysis": [
                {"source": "Social Media", "percentage": 45, "risk": 72},
                {"source": "Blogs", "percentage": 25, "risk": 58},
                {"source": "News Sites", "percentage": 15, "risk": 24},
                {"source": "Forums", "percentage": 10, "risk": 65},
                {"source": "Other", "percentage": 5, "risk": 40}
            ]
        }
    }

def get_topic_color(topic: str) -> str:
    """Map topic to frontend color"""
    color_map = {
        "vaccine": "hsl(0 72% 51%)",
        "cancer": "hsl(25 95% 53%)",
        "covid": "hsl(43 96% 50%)",
        "nutrition": "hsl(152 69% 40%)",
        "mental_health": "hsl(175 55% 32%)",
        "alternative_medicine": "hsl(200 70% 50%)",
        "treatment": "hsl(240 60% 50%)",
        "medication": "hsl(280 60% 50%)"
    }
    return color_map.get(topic, "hsl(0 0% 70%)")

@app.get("/api/history")
async def get_history(limit: int = 50):
    """Get analysis history for history.tsx"""
    try:
        recent = analysis_history[-limit:] if analysis_history else []
        
        return {
            "success": True,
            "history": [
                {
                    "id": h["id"],
                    "text": h["text"],
                    "source": h["source"],
                    "timestamp": h["timestamp"],
                    "result": h["result"],
                    "metrics": h.get("metrics", {})
                }
                for h in recent
            ],
            "count": len(recent),
            "total_count": len(analysis_history)
        }
    except Exception as e:
        raise HTTPException(500, f"History retrieval failed: {str(e)}")

@app.delete("/api/history")
async def clear_history():
    """Clear analysis history"""
    try:
        global analysis_history
        count = len(analysis_history)
        analysis_history = []
        
        return {
            "success": True,
            "message": f"Cleared {count} history entries",
            "cleared_count": count
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to clear history: {str(e)}")

@app.get("/api/topics")
async def get_topics():
    """Get supported health topics"""
    return {
        "success": True,
        "topics": HEALTH_TOPICS,
        "count": len(HEALTH_TOPICS),
        "description": "Health topics automatically detected by MedVerax AI"
    }

if __name__ == "__main__":
    import uvicorn
    
    # For Render deployment
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )