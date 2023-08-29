import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8888)) ,
        reload=False,
        log_level="info"
    )

