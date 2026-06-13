import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

USER_DATA_PATH = "user_submissions.csv"
ORIGINAL_DATA_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"
ARTIFACTS_PATH = "model_artifacts.joblib"

st.set_page_config(page_title="Obesity Prediction Dashboard", layout="wide")

# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("model_artifacts.joblib")

artifacts = load_artifacts()
models = artifacts["models"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]
results_df = artifacts["results_df"]
confusions = artifacts["confusions"]
class_labels = artifacts["class_labels"]
best_model_name = artifacts["best_model_name"]

st.title(" Obesity Level Prediction Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Model Performance", " Confusion Matrix", " Predict Your Obesity Class",
    " Collected Submissions", " Retrain Model"
])

# ------------------------------------------------------------------
# TAB 1: Model performance comparison
# ------------------------------------------------------------------
with tab1:
    st.subheader("Model Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=results_df, x="accuracy", y="model", palette="viridis", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("")
    for i, v in enumerate(results_df["accuracy"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    st.pyplot(fig)

    st.markdown(f"**Best performing model:** `{best_model_name}` "
                f"with an accuracy of **{results_df.iloc[0]['accuracy']:.2%}** on the test set.")

    st.dataframe(results_df.reset_index(drop=True), use_container_width=True)

# ------------------------------------------------------------------
# TAB 2: Confusion matrix
# ------------------------------------------------------------------
with tab2:
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox("Select a model", list(models.keys()),
                                   index=list(models.keys()).index(best_model_name))

    cm = confusions[selected_model]
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title(f"Confusion Matrix - {selected_model}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig2)

# ------------------------------------------------------------------
# TAB 3: Prediction form
# ------------------------------------------------------------------
with tab3:
    st.subheader("Enter Your Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=10, max_value=80, value=25)
        height = st.number_input("Height (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        family_history = st.selectbox("Family history of overweight?", ["yes", "no"])

    with col2:
        favc = st.selectbox("Frequent consumption of high caloric food (FAVC)?", ["yes", "no"])
        fcvc = st.slider("Vegetable consumption frequency (FCVC)", 1.0, 3.0, 2.0)
        ncp = st.slider("Number of main meals (NCP)", 1.0, 4.0, 3.0)
        caec = st.selectbox("Eating between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Do you smoke?", ["no", "yes"])

    with col3:
        ch2o = st.slider("Daily water intake (CH2O, liters)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Monitor calorie consumption (SCC)?", ["no", "yes"])
        faf = st.slider("Physical activity frequency (FAF)", 0.0, 3.0, 1.0)
        tue = st.slider("Time using technology devices (TUE, hours)", 0.0, 2.0, 1.0)
        calc = st.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation used (MTRANS)",
                               ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    predict_model_name = st.selectbox("Model to use for prediction", list(models.keys()),
                                       index=list(models.keys()).index(best_model_name))

    if st.button("Predict Obesity Class", type="primary"):
        input_dict = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)

        # Align columns with training feature columns
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)

        model = models[predict_model_name]
        prediction = model.predict(input_scaled)[0]

        bmi = weight / (height ** 2)

        st.success(f"### Predicted Obesity Class: **{prediction}**")
        st.info(f"Calculated BMI: **{bmi:.2f}**")

        # ------------------------------------------------------------
        # Save this submission to a CSV file for future retraining
        # ------------------------------------------------------------
        save_dict = dict(input_dict)
        save_dict["NObeyesdad"] = prediction  # keep same column name as original dataset
        save_dict["BMI"] = round(bmi, 2)
        save_dict["model_used"] = predict_model_name
        save_dict["timestamp"] = datetime.now().isoformat(timespec="seconds")

        save_df = pd.DataFrame([save_dict])

        if os.path.exists(USER_DATA_PATH):
            save_df.to_csv(USER_DATA_PATH, mode="a", header=False, index=False)
        else:
            save_df.to_csv(USER_DATA_PATH, mode="w", header=True, index=False)

        st.caption(f" Your input has been saved to `{USER_DATA_PATH}` for future model training.")


        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0]
            proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba}) \
                         .sort_values("Probability", ascending=False)
            st.subheader("Prediction Probabilities")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.barplot(data=proba_df, x="Probability", y="Class", palette="rocket", ax=ax3)
            ax3.set_xlim(0, 1)
            st.pyplot(fig3)

# ------------------------------------------------------------------
# TAB 4: View collected user submissions
# ------------------------------------------------------------------
with tab4:
    st.subheader("User-Submitted Data (for future retraining)")

    if os.path.exists(USER_DATA_PATH):
        collected_df = pd.read_csv(USER_DATA_PATH)
        st.write(f"Total submissions collected: **{len(collected_df)}**")
        st.dataframe(collected_df, use_container_width=True)

        csv_bytes = collected_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download user_submissions.csv",
            data=csv_bytes,
            file_name="user_submissions.csv",
            mime="text/csv",
        )
    else:
        st.info("No user submissions yet. Once someone uses the prediction tab, their data will appear here.")

# ------------------------------------------------------------------
# TAB 5: Retrain models using original + collected user data
# ------------------------------------------------------------------
with tab5:
    st.subheader("Retrain Models with Updated Data")

    st.markdown(
        "This will combine the original training dataset "
        f"(`{ORIGINAL_DATA_PATH}`) with all user-submitted entries "
        f"(`{USER_DATA_PATH}`), retrain all 5 models, and update the "
        "dashboard's saved model artifacts."
    )

    if not os.path.exists(ORIGINAL_DATA_PATH):
        st.error(f"Original dataset `{ORIGINAL_DATA_PATH}` not found in this folder. "
                 "Please place it alongside app.py to enable retraining.")
    else:
        original_df = pd.read_csv(ORIGINAL_DATA_PATH)
        n_original = len(original_df)

        n_user = 0
        if os.path.exists(USER_DATA_PATH):
            user_df = pd.read_csv(USER_DATA_PATH)
            n_user = len(user_df)

        st.write(f"Original dataset rows: **{n_original}**")
        st.write(f"User-submitted rows available: **{n_user}**")

        if st.button("🔄 Retrain Now", type="primary"):
            if n_user == 0:
                st.warning("No user-submitted data found yet. Retraining will use only the original dataset.")
                combined_df = original_df.copy()
            else:
                user_df = pd.read_csv(USER_DATA_PATH)
                # Keep only columns matching the original dataset structure
                user_df_clean = user_df.drop(columns=["BMI", "model_used", "timestamp"], errors="ignore")
                user_df_clean = user_df_clean.rename(columns={"NObeyesdad": "NObeyesdad"})
                combined_df = pd.concat([original_df, user_df_clean], ignore_index=True)

            with st.spinner("Retraining models, please wait..."):
                df = combined_df.rename(columns={"NObeyesdad": "ObesityPrediction"})
                y = df["ObesityPrediction"]
                x = df.drop(columns=["ObesityPrediction"])
                x = pd.get_dummies(x)

                new_feature_columns = x.columns.tolist()

                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=42, stratify=y
                )

                new_scaler = StandardScaler()
                x_train_s = new_scaler.fit_transform(x_train)
                x_test_s = new_scaler.transform(x_test)

                model_defs = {
                    "GaussianNB": GaussianNB(),
                    "SVC": SVC(probability=True),
                    "RandomForestClassifier": RandomForestClassifier(random_state=42),
                    "LogisticRegression": LogisticRegression(max_iter=2000),
                    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
                }

                new_results = []
                new_trained_models = {}
                new_confusions = {}
                new_class_labels = sorted(y.unique())

                for name, model in model_defs.items():
                    model.fit(x_train_s, y_train)
                    preds = model.predict(x_test_s)
                    acc = accuracy_score(y_test, preds)
                    cm = confusion_matrix(y_test, preds, labels=new_class_labels)
                    new_results.append({"model": name, "accuracy": acc})
                    new_trained_models[name] = model
                    new_confusions[name] = cm

                new_results_df = pd.DataFrame(new_results).sort_values("accuracy", ascending=False)
                new_best_name = new_results_df.iloc[0]["model"]

                joblib.dump({
                    "models": new_trained_models,
                    "scaler": new_scaler,
                    "feature_columns": new_feature_columns,
                    "results_df": new_results_df,
                    "confusions": new_confusions,
                    "class_labels": new_class_labels,
                    "best_model_name": new_best_name,
                }, ARTIFACTS_PATH)

            st.success(" Retraining complete! Model artifacts updated.")
            st.write(f"New training set size: **{len(x_train)} rows** "
                     f"(total combined dataset: **{len(combined_df)} rows**)")
            st.dataframe(new_results_df.reset_index(drop=True), use_container_width=True)
            st.info(f"New best model: **{new_best_name}**")

            st.warning("Reload the app (press 'R' or refresh the page) to see the updated "
                       "results in the Model Performance and Confusion Matrix tabs.")
