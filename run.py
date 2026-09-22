from app import app
from config import config
if __name__ == "__main__":
    print("Smart Assistant: http://{}:{}".format(config.HOST, config.PORT))
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, load_dotenv=False)
