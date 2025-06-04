from pyngrok import ngrok, conf
import uvicorn
import nest_asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set up path to ensure imports work properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Securely set Ngrok auth token
conf.get_default().auth_token = os.getenv("NGROK_AUTH_TOKEN", "your-fallback-token-here")

# Start Ngrok tunnel
public_url = ngrok.connect(8000)
print("🔗 Public URL:", public_url)

# Apply nest_asyncio to allow nested event loops (for Colab or Jupyter)
nest_asyncio.apply()

# ✅ Gracefully run the FastAPI server
try:
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000)
except KeyboardInterrupt:
    print("🛑 Shutting down gracefully...")
