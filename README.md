# 🏠 Bucharest Real Estate Price Prediction (AI-Housing)

This project is a **full-cycle Machine Learning** application developed to estimate real-estate property values in Bucharest. It integrates a complex neural network with a modern graphical interface, giving users a precise market-analysis tool based on real technical parameters and current 2024–2025 trends.

---

## 🎯 1. Project Goal and Objectives
The project was designed as an Artificial Intelligence solution for the real-estate market, aiming to eliminate subjectivity in property valuation.

* **Accurate Estimation with Deep Learning:** Using neural networks to predict apartment prices based on location (district), surface area, and amenities.
* **Statistical Analysis:** Visualizing correlations between variables (e.g., the influence of metro distance on the final price).
* **Intuitive Interface:** Providing a GUI (Graphical User Interface) tool that lets the user get instant predictions without writing code.

---

## ⚙️ 2. Technologies and Tools Used

The application uses a modern technology stack specific to Data Science and AI:

* **Programming Language:**
  * **Python:** The main language used for data processing, model training, and UI logic.
* **Machine Learning & AI:**
  * **TensorFlow & Keras:** The neural network architecture for numerical regression.
  * **Scikit-learn:** Used for data preprocessing (`StandardScaler`) and evaluation metrics.
* **Data Analysis & Visualization:**
  * **Pandas & NumPy:** Dataset manipulation and matrix computations.
  * **Matplotlib & Seaborn:** Generating charts for error analysis and price distribution.
* **Graphical Interface:**
  * **Tkinter:** Framework for building the interactive desktop window.

---

## 🧠 3. System Architecture (AI Deep Dive)
The project is based on the `saved_model.keras` model and its associated processing script:

* **Input Features:** The model processes 8 key variables: `District`, `Surface Area`, `No. of Rooms`, `Floor`, `Construction Year`, `Metro Distance`, `No. of Bathrooms`, and `Balcony`.
* **Neural Architecture:** A **Sequential** network with dense hidden layers and `ReLU` activation, optimized with the `Adam` algorithm and `L2` regularization to prevent overfitting.
* **Data Normalization:** Using `scaler.pkl` to bring all inputs into the same numerical range — essential for the stability of the neural network.
* **Validation:** The model was tested on extreme scenarios (from budget apartments in District 6 to ultra-premium homes in District 1).

---

## 🚀 4. Detailed Features

### A. Price Predictor
* **Dynamic Input:** A complex form with validation for the property's technical data.
* **Dual Display:** Automatic price calculation in both **Euro** and **RON** (at the updated exchange rate).

### B. Visualization & Analysis Module
* **Performance Charts:** Automatic generation of "Actual vs Predicted" charts to demonstrate the model's accuracy.
* **Data Export:** The ability to export the prediction history to `.csv`.

### C. Reliability & Stability
* **Error Handling:** An alert system (Messagebox) for invalid inputs or missing model files.
* **Feature Synchronization:** Using `feature_names.json` to ensure the correct order of data fed into the model.

---

## 🛠️ 5. Installation & Running Instructions
1. Install **Python 3.10** or newer.
2. Install the required dependencies:
   ```bash
   pip install tensorflow pandas scikit-learn matplotlib seaborn
   ```
3. Place the `saved_model.keras`, `scaler.pkl`, and `feature_names.json` files in the same directory as the main script.
4. Run the application:
   ```bash
   python house_price_predictor_enhanced.py
   ```
5. Enter the property details in the interface and click "Calculate Estimated Price".

---

## 👨‍💻 6. Author
Project by **Nicolae-Bogdan Proaspătu**
*Automation and Applied Informatics student, Technical University of Civil Engineering Bucharest*
Academic year 2024–2025.

---

## ⚖️ 7. License
Developed for academic purposes as part of the **Artificial Intelligence** course.
