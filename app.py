@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    clinical_note = ""
    
    if request.method == "POST":
        uploaded_files = request.files.getlist("mat_files")
        if uploaded_files:
            static_dir = os.path.join(BASE_DIR, "static")
            if not os.path.exists(static_dir): os.makedirs(static_dir)
            
            predicted_levels = []
            filenames = []

            for file in uploaded_files:
                file_path = os.path.join(static_dir, file.filename)
                file.save(file_path)
                filenames.append(file.filename)
                
                try:
                    data = loadmat(file_path)
                    eeg = data.get("Data")
                    
                    if stress_model is None:
                        raise ValueError("Model file (stress_model.pkl) not loaded.")

                    # 1. Extract features using the Colab-mirror function
                    feats = extract_features(eeg)
                    
                    # 2. Get Raw Probabilities (Just like your probs = model.predict_proba line)
                    prob = stress_model.predict_proba(feats)[:, 1][0]
                    
                    # 3. Apply the GOLDEN THRESHOLD (0.6)
                    # This is what gives you the 82% accuracy
                    state = "Stress" if prob > 0.6 else "Relax"
                    
                    # Add result with percentage for your own tracking
                    predicted_levels.append(f"{state} ({int(prob*100)}%)")
                    
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")
                    predicted_levels.append("Error")

            if predicted_levels:
                # Count only the 'Stress' and 'Relax' parts for the summary
                just_labels = [p.split(" ")[0] for p in predicted_levels]
                counts = Counter(just_labels)
                main_state = counts.most_common(1)[0][0]

                # Gemini AI Report
                try:
                    model_gen = genai.GenerativeModel('gemini-2.0-flash-lite')
                    prompt = f"EEG Analysis: {dict(counts)}. Majority: {main_state}. Write a short medical summary."
                    response = model_gen.generate_content(prompt)
                    clinical_note = response.text.strip().replace("\n\n", "<br><br>")
                except:
                    clinical_note = f"Session complete. Majority state: {main_state}."

                # Results for the table
                results = list(zip(filenames, predicted_levels))
            
    return render_template("index.html", results=results, clinical_note=clinical_note)
