
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

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any
import google.generativeai as genai
import logging
import os
import json
import re

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medverax-backend")

# ---------------- APP ----------------
app = FastAPI(
    title="MedVerax AI Backend",
    description="AI-powered Health Misinformation Detection System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- GEMINI SETUP ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini AI connected")
    except Exception as e:
        logger.warning(f"Gemini init failed: {e}")

GEMINI_AVAILABLE = gemini_model is not None

# ---------------- CONSTANTS ----------------
RISK_LEVELS = {
    "low": "Likely reliable",
    "medium": "Potentially misleading",
    "high": "Probably misinformation",
    "critical": "Dangerous misinformation"
}

ANALYSIS_HISTORY = []
MAX_HISTORY = 500

# ---------------- UTILITIES ----------------
def detect_topics(text: str) -> List[str]:
    topics = {
        "covid": ["covid", "coronavirus"],
        "cancer": ["cancer", "tumor"],
        "vaccine": ["vaccine", "vaccination"],
        "mental_health": ["depression", "anxiety"],
        "nutrition": ["diet", "nutrition"],
        "medicine": ["medicine", "drug", "treatment"]
    }
    found = []
    t = text.lower()
    for k, v in topics.items():
        if any(word in t for word in v):
            found.append(k)
    return found if found else ["general_health"]

def rule_based_analysis(text: str) -> Dict[str, Any]:
    t = text.lower()
    misinfo = any(w in t for w in ["miracle", "100%", "secret cure", "they don't want you to know"])
    confidence = 85 if misinfo else 70
    return {
        "is_misinformation": misinfo,
        "confidence_score": confidence,
        "risk_level": "high" if misinfo else "low",
        "verdict": "Health misinformation detected" if misinfo else "Likely reliable health information",
        "explanation": "Detected exaggerated or unverified medical claims." if misinfo else "Uses cautious medical language.",
        "key_issues": ["Absolute claims"] if misinfo else [],
        "evidence_needed": ["WHO / CDC guidelines", "Peer-reviewed research"],
        "recommended_action": "Consult healthcare professionals"
    }

def gemini_analysis(text: str, topics: List[str]) -> Dict[str, Any]:
    prompt = f"""
    Analyze this health content for misinformation.

    Text: "{text}"
    Topics: {topics}

    Respond ONLY in JSON:
    {{
      "is_misinformation": true/false,
      "confidence_score": 0-100,
      "risk_level": "low|medium|high|critical",
      "verdict": "",
      "explanation": "",
      "key_issues": [],
      "evidence_needed": [],
      "recommended_action": ""
    }}
    """
    response = gemini_model.generate_content(prompt)
    cleaned = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(cleaned)

def calculate_metrics(text: str, base: Dict[str, Any]) -> Dict[str, Any]:
    urgency = sum(w in text.lower() for w in ["urgent", "warning", "immediately"])
    sensational = sum(w in text.lower() for w in ["miracle", "shocking"])
    return {
        "pattern_strengths": {
            "urgency": urgency * 20,
            "sensationalism": sensational * 25
        },
        "overall_risk_score": base["confidence_score"]
    }

# ---------------- ENDPOINTS ----------------
@app.get("/")
def root():
    return {
        "service": "MedVerax AI Backend",
        "status": "active",
        "gemini_available": GEMINI_AVAILABLE
    }

@app.post("/analyze")
def analyze(
    text: str = Form(...),
    use_ai: bool = Form(True)
):
    if len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short")

    topics = detect_topics(text)

    if use_ai and GEMINI_AVAILABLE:
        analysis = gemini_analysis(text, topics)
        source = "Gemini AI"
    else:
        analysis = rule_based_analysis(text)
        source = "Rule-Based"

    metrics = calculate_metrics(text, analysis)

    result = {
        "analysis": {
            "text_preview": text[:200],
            "topics": topics,
            **analysis,
            "analysis_source": source,
            "timestamp": datetime.utcnow().isoformat()
        },
        "metrics": metrics
    }

    ANALYSIS_HISTORY.append(result["analysis"])
    if len(ANALYSIS_HISTORY) > MAX_HISTORY:
        ANALYSIS_HISTORY.pop(0)

    return result

@app.get("/history")
def history(limit: int = 20):
    return ANALYSIS_HISTORY[-limit:]

@app.get("/analytics/summary")
def analytics():
    risks = {k: 0 for k in RISK_LEVELS}
    for h in ANALYSIS_HISTORY:
        risks[h["risk_level"]] += 1
    return {
        "total": len(ANALYSIS_HISTORY),
        "risk_distribution": risks
    }

@app.post("/simulate")
def simulate(
    base_risk: int = Form(...),
    source: str = Form("unknown")
):
    risk = base_risk
    if source == "trusted":
        risk -= 10
    elif source == "unverified":
        risk += 10
    return {
        "original_risk": base_risk,
        "simulated_risk": max(0, min(100, risk)),
        "note": "Simulation only"
    }

@app.get("/model/info")
def model_info():
    return {
        "model": "Gemini AI + Rule-Based Fallback",
        "version": "2.0.0",
        "explainable": True,
        "analytics_supported": True
    }
