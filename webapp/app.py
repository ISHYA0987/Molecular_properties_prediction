import sys
import os
from flask import Flask, render_template, request

# 🔥 Ensure project root is in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_from_smiles

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        smiles = request.form.get("smiles", "").strip()

        # 🔥 Input validation
        if not smiles:
            return render_template("index.html", error="Please enter a SMILES string.")

        try:
            result = predict_from_smiles(smiles)

            # 🔥 Handle prediction errors
            if result is None:
                return render_template("index.html", error="Prediction failed.")

            if "error" in result:
                return render_template("index.html", error=result["error"])

            return render_template("result.html", result=result)

        except Exception as e:
            # 🔥 Debug print (check terminal)
            print("ERROR:", str(e))

            return render_template(
                "index.html",
                error="Something went wrong. Please try again."
            )

    return render_template("index.html")


# 🔥 Run server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)