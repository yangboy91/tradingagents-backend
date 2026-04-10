import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储任务结果
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
class AnalysisRequest(BaseModel):
    ticker: str
    date: str

def run_analysis(task_id: str, ticker: str, date: str):
    try:
        tasks[task_id] = {"status": "running", "result": None}
        ta = TradingAgentsGraph(config=config)
        final_state, decision = ta.propagate(ticker, date)
        tasks[task_id] = {
            "status": "done",
            "result": {
                "decision": decision,
                "market_report": final_state.get("market_report", ""),
                "news_report": final_state.get("news_report", ""),
                "fundamentals_report": final_state.get("fundamentals_report", ""),
                "sentiment_report": final_state.get("sentiment_report", ""),
            }
        }
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
