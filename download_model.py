import gdown
import os

def download_model():
    file_id = "1K4xjoke4u7mP9oUcbIBF7qtognNB9I98"
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path = "Diabetes/Diabetes_model.pkl"

    # Create folder if missing
    os.makedirs("Diabetes", exist_ok=True)

    # Download if not present
    if not os.path.exists(out_path):
        print("Downloading diabetes model from Google Drive...")
        gdown.download(url, out_path, quiet=False)
        print("Download completed:", out_path)
    else:
        print("Model already exists:", out_path)
