import os

import uvicorn

from app.api import create_app

app = create_app()


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
