"""
HackRx LLM-Powered Query Retrieval System
Optimized for insurance, legal, HR, and compliance domains
"""

import uvicorn
from src.api.main import app

if __name__ == "__main__":
    print("🚀 Starting HackRx LLM Query Retrieval System...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        access_log=True
    )
