import uvicorn

from common.config import GEMINI_HOST, GEMINI_PORT
from gemini_node.server import build_app

if __name__ == "__main__":
    app = build_app()
    uvicorn.run(app.build(), host=GEMINI_HOST, port=GEMINI_PORT)
