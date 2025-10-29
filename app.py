import os
from dotenv import load_dotenv


def main():
    load_dotenv()  # this gives us KAGGLE_USERNAME and KAGGLE_KEY
    os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

    print(os.environ["KERAS_BACKEND"])


if __name__ == "__main__":
    main()
