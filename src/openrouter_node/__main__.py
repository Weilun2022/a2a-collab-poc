import uvicorn

from common.config import OPENROUTER_HOST, OPENROUTER_PORT
from openrouter_node.server import build_app

if __name__ == "__main__":
    app = build_app()
    uvicorn.run(app.build(), host=OPENROUTER_HOST, port=OPENROUTER_PORT)
