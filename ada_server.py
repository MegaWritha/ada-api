from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# === VOCABULARY ===
vocabulary = [
    "love", "joy", "wonderful", "celebrate", "happy", "laugh",
    "dance", "sing", "bright", "beautiful", "smile", "delight",
    "proud", "strong", "brave", "mighty", "glory", "heritage",
    "culture", "ancient", "wise", "sacred", "honour", "victory",
    "hope", "free", "rise", "future", "dream", "light",
    "grow", "build", "new", "chance", "begin", "believe",
    "cry", "tears", "mourn", "lost", "gone", "miss",
    "death", "alone", "empty", "silence", "ache", "sorrow",
    "anger", "rage", "injustice", "wrong", "fight", "resist",
    "oppression", "suffer", "pain", "broken", "war", "destroy",
    "fear", "dark", "danger", "threat", "hide", "run",
    "terror", "weak", "trap", "enemy", "cold", "shadow",
    "africa", "nigeria", "woman", "man", "child", "land",
    "river", "market", "village", "city", "family", "home",
    "book", "story", "voice", "heart", "mind", "soul"
]

vocabulary = list(dict.fromkeys(vocabulary))

def vectorize(sentence):
    words = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for word in words:
        if word in vocabulary:
            index = vocabulary.index(word)
            vector[index] = 1
    return vector

# === TRAINING DATA ===
emotions = {
    "joy": {
        "sentences": [
            "we laugh and dance and celebrate together",
            "her smile brought joy to the whole village",
            "a beautiful bright day full of delight",
            "the children sing and play with happiness",
        ],
        "answers": [1, 1, 1, 1]
    },
    "pride": {
        "sentences": [
            "proud of our ancient culture and heritage",
            "the brave and mighty warriors of africa",
            "our sacred land full of glory and honour",
            "wise elders carry the victory of our people",
        ],
        "answers": [1, 1, 1, 1]
    },
    "hope": {
        "sentences": [
            "we will rise and build a new future",
            "the dream of a free africa begins today",
            "light grows in the hearts of those who believe",
            "a new chance to begin again",
        ],
        "answers": [1, 1, 1, 1]
    },
    "grief": {
        "sentences": [
            "she cried alone in the silence missing the gone",
            "tears and sorrow for the lost ones",
            "the emptiness after death leaves an ache",
            "mourning the ones who are no longer here",
        ],
        "answers": [1, 1, 1, 1]
    },
    "anger": {
        "sentences": [
            "rage against the injustice and oppression",
            "we fight and resist the broken system",
            "pain and suffering from a long war",
            "anger at the wrong that destroys our people",
        ],
        "answers": [1, 1, 1, 1]
    },
    "fear": {
        "sentences": [
            "hiding in the dark from the enemy",
            "terror and shadow make the weak run",
            "trapped by danger with nowhere to go",
            "cold fear of the threat that surrounds us",
        ],
        "answers": [1, 1, 1, 1]
    },
}

# === TRAIN ===
all_weights = {}

for emotion, data in emotions.items():
    weights = np.zeros(len(vocabulary))
    learning_rate = 0.1
    for round in range(200):
        for i in range(len(data["sentences"])):
            vector = vectorize(data["sentences"][i])
            prediction = np.dot(vector, weights)
            error = data["answers"][i] - prediction
            weights = weights + learning_rate * error * vector
    all_weights[emotion] = weights

print("Ada v3 is trained and ready!")

# === CLASSIFY ===
def classify(sentence):
    vector = vectorize(sentence)
    scores = {}
    for emotion, weights in all_weights.items():
        scores[emotion] = float(np.dot(vector, weights))
    top_emotion = max(scores, key=scores.get)
    return top_emotion, scores

# === ROUTES ===
@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json()
    sentence = data.get("text", "")
    emotion, scores = classify(sentence)
    return jsonify({
        "text": sentence,
        "emotion": emotion,
        "scores": scores
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Ada v3 is running"})

import requests as req

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    HF_KEY = os.environ.get("HF_API_KEY", "")
    
    response = req.post(
        "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
        headers={"Authorization": f"Bearer {HF_KEY}"},
        json={"inputs": prompt}
    )
    
    if response.ok:
        import base64
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return jsonify({"image": f"data:image/jpeg;base64,{image_base64}"})
    else:
        error_text = response.text
        print(f"HF Error {response.status_code}: {error_text}")
        return jsonify({"error": "Image generation failed", "details": error_text}), 500            

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)