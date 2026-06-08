"""
app.py - Simple GUI for Emotion Recognition
Run: python app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import os
from dataset import load_audio, extract_features
from model import build_model

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognizer")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        # Create all GUI widgets
        self.label_title = tk.Label(root, text="Emotion Recognition from Speech", font=("Time new Roman", 14, "bold"))
        self.label_title.pack(pady=10)

        self.btn_select = tk.Button(root, text="Select Audio File (.wav)", command=self.select_file, font=("Time new Roman", 10), width=30)
        self.btn_select.pack(pady=5)

        self.label_file = tk.Label(root, text="No file selected", fg="gray")
        self.label_file.pack(pady=5)

        self.btn_predict = tk.Button(root, text="Predict Emotion", command=self.predict, state=tk.DISABLED, font=("Time new Roman", 10), width=20)
        self.btn_predict.pack(pady=10)

        self.label_result = tk.Label(root, text="", font=("Time new Roman", 12, "bold"))
        self.label_result.pack(pady=10)

        self.label_confidence = tk.Label(root, text="", font=("Time new Roman", 10))
        self.label_confidence.pack(pady=5)

        self.progress = tk.Label(root, text="", fg="blue")
        self.progress.pack(pady=5)

        self.file_path = None

        # Load model
        self.model = None
        self.classes = None
        self.device = torch.device("cpu")
        self.load_model()   # <-- calls the method defined below

    def load_model(self):
        """Load the trained model from checkpoints/best_model.pt"""
        checkpoint_path = "checkpoints/best_model.pt"
        if not os.path.exists(checkpoint_path):
            messagebox.showerror("Error", "Model checkpoint not found!\nTrain the model first using main.py")
            self.root.quit()
            return

        try:
            self.progress.config(text="Loading model...")
            self.root.update()

            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.classes = ckpt["classes"]
            num_classes = len(self.classes)

            self.model = build_model(model_type="cnn", num_classes=num_classes)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self.device)
            self.model.eval()

            self.progress.config(text="Model loaded. Ready.")
            self.root.update()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.root.quit()

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if self.file_path:
            self.label_file.config(text=f"Selected: {os.path.basename(self.file_path)}", fg="black")
            self.btn_predict.config(state=tk.NORMAL)
            self.label_result.config(text="")
            self.label_confidence.config(text="")
        else:
            self.label_file.config(text="No file selected", fg="gray")
            self.btn_predict.config(state=tk.DISABLED)

    def predict(self):
        if not self.file_path:
            return

        try:
            self.progress.config(text="Processing audio...")
            self.root.update()

            # Load and extract features
            y = load_audio(self.file_path)
            feat = extract_features(y)                     # shape (53, MAX_LEN)
            tensor = torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,53,T)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            pred_idx = probs.argmax()
            pred_label = self.classes[pred_idx]
            confidence = probs[pred_idx] * 100

            self.label_result.config(text=f"Predicted Emotion: {pred_label.upper()}", fg="green")
            self.label_confidence.config(text=f"Confidence: {confidence:.1f}%")
            self.progress.config(text="Done.")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.progress.config(text="Error occurred")
        finally:
            self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()