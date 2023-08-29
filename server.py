import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=os.environ["HOST"] or "127.0.0.1",
        port=int(os.environ["PORT"]) or 8888,
        reload=False,
        log_level="info"
    )

