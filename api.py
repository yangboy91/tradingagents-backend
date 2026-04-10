import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from openai import OpenAI

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks = {}

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-4o"
config["quick_think_llm"] = "gpt-4o-mini"
config["data_vendors"] = {
    "core_stock_apis": "alpha_vantage",
    "technical_indicators": "alpha_vantage",
    "fundamental_data": "alpha_vantage",
    "news_data": "alpha_vantage",
}

client = OpenAI()

def structure_report(ticker: str, date: str, raw: dict) -> dict:
    prompt = f"""You are a senior buy-side equity analyst. Based on the following raw research inputs for {ticker} on {date}, produce a structured research note.

RAW INPUTS:
=== TECHNICAL ANALYSIS ===
{raw.get('market_report', '')}

=== FUNDAMENTALS ===
{raw.get('fundamentals_report', '')}

=== NEWS & MACRO ===
{raw.get('news_report', '')}

=== MARKET SENTIMENT ===
{raw.get('sentiment_report', '')}

=== FINAL RECOMMENDATION ===
{raw.get('decision', '')}

Produce a professional research note with EXACTLY these sections:

## Executive Summary
2-3 sentences: what is the recommendation and the single most important reason.

## Investment Thesis
3-5 bullet points explaining WHY this is a {raw.get('decision', '')} with specific evidence from the data.

## Key Financials
A brief summary of the most important financial metrics (revenue, margins, growth, valuation multiples). Use actual numbers if available.

## Technical Picture
Current price trend, key support/resistance levels, and what the indicators say.

## Risk Factors
3-5 specific risks that could invalidate the thesis. Be honest and specific.

## Catalysts to Watch
2-3 upcoming events or data points that could move the stock.

## Recommendation
You MUST end with exactly one of these three words on its own line: BUY, HOLD, or SELL. No other variations like "Overweight" or "Underweight" are acceptable.

Write in a professional, direct tone. Use specific numbers and evidence. Avoid vague language."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    structured = response.choices[0].message.content

    # Normalize decision to BUY/HOLD/SELL
    decision_raw = raw.get("decision", "HOLD").upper()
    if any(w in decision_raw for w in ["BUY", "OVERWEIGHT", "LONG"]):
        normalized_decision = "BUY"
    elif any(w in decision_raw for w in ["SELL", "UNDERWEIGHT", "SHORT"]):
        normalized_decision = "SELL"
    else:
        normalized_decision = "HOLD"

    return {
        "decision": normalized_decision,
        "structured_report": structured,
        "market_report": raw.get("market_report", ""),
        "news_report": raw.get("news_report", ""),
        "fundamentals_report": raw.get("fundamentals_report", ""),
        "sentiment_report": raw.get("sentiment_report", ""),
    }

class AnalysisRequest(BaseModel):
    ticker: str
    date: str

def run_analysis(task_id: str, ticker: str, date: str):
    try:
        tasks[task_id] = {"status": "running", "result": None}
        ta = TradingAgentsGraph(config=config)
        final_state, decision = ta.propagate(ticker, date)

        raw = {
            "decision": decision,
            "market_report": final_state.get("market_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
        }

        structured = structure_report(ticker, date, raw)
        tasks[task_id] = {"status": "done", "result": structured}
    except Exception as e:
        tasks[task_id] = {"status": "error", "result": str(e)}

@app.post("/analyze")
async def analyze(req: AnalysisRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "result": None}
    background_tasks.add_task(run_analysis, task_id, req.ticker, req.date)
    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks:
        return {"status": "not_found"}
    return tasks[task_id]

@app.get("/health")
async def health():
    return {"status": "ok"}
