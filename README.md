# 🏠 Predicție Prețuri Imobiliare București (AI-Housing)

Acest proiect reprezintă o aplicație de tip **Machine Learning Full-Cycle** dezvoltată pentru a estima valorile proprietăților imobiliare din București. Aplicația integrează o rețea neuronală complexă cu o interfață grafică modernă, oferind utilizatorilor un instrument precis de analiză a pieței bazat pe parametri tehnici reali și tendințe actuale din 2024-2025.

---

## 🎯 1. Scopul și Obiectivele Proiectului
Proiectul a fost conceput ca o soluție bazată pe Inteligență Artificială pentru piața imobiliară, având ca scop eliminarea subiectivității în evaluarea locuințelor.

* **Estimare Precisă prin Deep Learning:** Utilizarea rețelelor neuronale pentru a prezice prețul apartamentelor în funcție de locație (sector), suprafață și facilități.
* **Analiză Statistică:** Vizualizarea corelațiilor dintre variabile (ex: influența distanței față de metrou asupra prețului final).
* **Interfață Intuitivă:** Oferirea unui instrument de tip GUI (Graphical User Interface) care permite utilizatorului să obțină predicții instantanee fără a scrie cod.

---

## ⚙️ 2. Tehnologii și Instrumente Utilizate

Aplicația utilizează o stivă tehnologică modernă specifică domeniului Data Science și AI:

* **Limbaj de Programare:**
  * **Python:** Limbajul principal utilizat pentru procesarea datelor, antrenarea modelului și logica UI.
* **Machine Learning & AI:**
  * **TensorFlow & Keras:** Arhitectura rețelei neuronale pentru regresie numerică.
  * **Scikit-learn:** Utilizat pentru preprocesarea datelor (`StandardScaler`) și metrici de evaluare.
* **Analiza Datelor & Vizualizare:**
  * **Pandas & NumPy:** Manipularea seturilor de date și calcule matriceale.
  * **Matplotlib & Seaborn:** Generarea graficelor pentru analiza erorilor și a distribuției prețurilor.
* **Interfață Grafică:**
  * **Tkinter:** Framework pentru dezvoltarea ferestrei interactive de tip Desktop.

---

## 🧠 3. Arhitectura Sistemului (AI Deep Dive)
Proiectul se bazează pe modelul `saved_model.keras` și scriptul de procesare aferent:

* **Input Features:** Modelul procesează 8 variabile cheie: `Sector`, `Suprafață`, `Nr. Camere`, `Etaj`, `An Construcție`, `Distanță Metrou`, `Nr. Băi` și `Balcon`.
* **Arhitectură Neuronală:** Rețea de tip **Sequential** cu straturi dense (Hidden Layers) și activare de tip `ReLU`, optimizată cu algoritmul `Adam` și regularizare `L2` pentru prevenirea overfitting-ului.
* **Normalizarea Datelor:** Utilizarea `scaler.pkl` pentru aducerea tuturor input-urilor în același interval numeric, esențial pentru stabilitatea rețelei neuronale.
* **Validare:** Modelul a fost testat pe scenarii extreme (de la apartamente tip budget în Sector 6 până la locuințe ultra-premium în Sector 1).

---

## 🚀 4. Funcționalități Detaliate

### A. Predictor de Preț
* **Input Dinamic:** Formular complex cu validări pentru datele tehnice ale imobilului.
* **Dublă Afișare:** Calculul automat al prețului în **Euro** și **RON** (la curs actualizat).

### B. Modul de Vizualizare & Analiză
* **Grafice de Performanță:** Generarea automată a graficelor "Actual vs Predicted" pentru a demonstra acuratețea modelului.
* **Export Date:** Posibilitatea de a exporta istoricul predicțiilor efectuate în format `.csv`.

### C. Securitate și Stabilitate
* **Gestionarea Erorilor:** Sistem de alertă (Messagebox) pentru input-uri invalide sau lipsa fișierelor de model.
* **Sincronizarea Caracteristicilor:** Utilizarea `feature_names.json` pentru a asigura ordinea corectă a datelor la intrarea în model.

---

## 🛠️ 5. Instrucțiuni de Instalare și Rulare
1. Instalați **Python 3.10** sau o versiune mai nouă.
2. Instalați dependințele necesare:
   ```bash
   pip install tensorflow pandas scikit-learn matplotlib seaborn
   Plasați fișierele saved_model.keras, scaler.pkl și feature_names.json în același director cu scriptul principal.

3.Rulați aplicația:
 ```bash
python house_price_predictor_enhanced.py 
```
4. Introduceți detaliile imobilului în interfață și apăsați "Calculează Preț Estimativ".

## 👨‍💻 6.Realizat de
Proiect realizat de Nicolae-Bogdan Proaspătu Student la Automatică și Informatică Aplicată, Universitatea Tehnică de Construcții București. An universitar 2024–2025.

## ⚖️ 7. Licenta
Acest proiect este dezvoltat în scop academic pentru disciplina Tehnologii Web.
