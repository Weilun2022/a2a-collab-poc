import uvicorn

from claude_node.server import build_app
from common.config import CLAUDE_HOST, CLAUDE_PORT

if __name__ == "__main__":
    app = build_app()
    uvicorn.run(app.build(), host=CLAUDE_HOST, port=CLAUDE_PORT)
