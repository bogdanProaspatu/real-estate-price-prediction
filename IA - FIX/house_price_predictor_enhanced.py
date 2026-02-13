import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import pickle
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

class ModernButton(tk.Canvas):
    """Custom modern button widget with animations"""
    def __init__(self, parent, text, command, bg_color, fg_color, hover_color, **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.hover_color = hover_color
        self.text = text
        self.is_hovered = False
        
        self.bind('<Button-1>', lambda e: self.command())
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
        self.draw_button()
        
    def draw_button(self, color=None):
        if color is None:
            color = self.bg_color
        self.delete('all')
        width = self.winfo_reqwidth() if self.winfo_reqwidth() > 1 else 200
        height = self.winfo_reqheight() if self.winfo_reqheight() > 1 else 40
        
        # Rounded rectangle effect
        self.create_rectangle(2, 2, width-2, height-2,
                            fill=color, outline='', tags='bg')
        self.create_text(width//2, height//2,
                        text=self.text, fill=self.fg_color, 
                        font=('Segoe UI', 11, 'bold'))
        
    def on_enter(self, e):
        self.is_hovered = True
        self.draw_button(self.hover_color)
        
    def on_leave(self, e):
        self.is_hovered = False
        self.draw_button(self.bg_color)

class HousePricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 Prețuri Imobiliare București")
        self.root.geometry("1600x1000")
        self.root.state('zoomed')  # Maximized window
        
        # Professional color scheme - White Dominant with Red Accents
        self.colors = {
            'bg_dark': '#f8f9fa',        # Almost white
            'bg_medium': '#ffffff',      # Pure white
            'bg_light': '#e9ecef',       # Very light gray
            'accent': '#dc3545',         # Bootstrap red
            'accent_hover': '#c82333',   # Dark red
            'success': '#28a745',        # Bootstrap green
            'warning': '#ffc107',        # Bootstrap amber
            'danger': '#dc3545',         # Bootstrap red
            'text': '#212529',           # Almost black
            'text_secondary': '#6c757d', # Gray
            'card': '#ffffff',           # Pure white cards
            'input_bg': '#f8f9fa',       # Light gray
            'purple': '#6f42c1',         # Purple
            'pink': '#e83e8c',           # Pink
            'orange': '#fd7e14',         # Orange
            'blue': '#007bff',           # Bootstrap blue
            'green': '#28a745',          # Green
            'teal': '#20c997',           # Teal
            'yellow': '#ffc107',         # Yellow
            'red_light': '#f8d7da',      # Light red background
            'red_medium': '#dc3545',     # Medium red
            'red_dark': '#bd2130',       # Dark red
            'border': '#dee2e6'          # Light border
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Variables
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.history = None
        self.data = None
        self.prediction_history = []
        self.model_path = 'saved_model.keras'
        self.data_path = 'synthetic_data.csv'
        self.scaler_path = 'scaler.pkl'
        self.inputs = {}
        
        # Statistics
        self.sector_stats = {}
        self.market_trends = {}
        
        # Create UI
        self.create_ui()
        
        # Load or train model
        self.load_or_train_model()
        
    def generate_synthetic_data(self, n_samples=2000):
        """Generează date sintetice realiste pentru piața imobiliară din București"""
        np.random.seed(42)
        
        sectors = np.random.randint(1, 7, n_samples)
        area = np.random.normal(70, 25, n_samples)
        area = np.clip(area, 30, 200)
        rooms = np.random.randint(1, 6, n_samples)
        floor = np.random.randint(0, 11, n_samples)
        year_built = np.random.randint(1970, 2025, n_samples)
        metro_distance = np.random.exponential(1.5, n_samples)
        metro_distance = np.clip(metro_distance, 0.1, 5)
        
        # Bathrooms - realistic distribution: mostly 1 bathroom, some have 2
        bathrooms = np.random.choice([1, 2], n_samples, p=[0.65, 0.35])
        
        balcony = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # REALISTIC PRICE CALCULATION - Based on actual Bucharest market 2024
        
        # Base price per sqm varies by sector (realistic market values)
        base_price_per_sqm = np.where(sectors == 1, 2200,      # Sector 1: Premium areas
                             np.where(sectors == 2, 1900,      # Sector 2: High-end
                             np.where(sectors == 3, 1600,      # Sector 3: Good
                             np.where(sectors == 4, 1400,      # Sector 4: Average
                             np.where(sectors == 5, 1250,      # Sector 5: Below average
                             1150)))))                          # Sector 6: Lower end
        
        # Age impact - MAJOR factor
        age = 2024 - year_built
        age_multiplier = np.where(age < 3, 1.25,               # Brand new (2022-2024)
                         np.where(age < 8, 1.15,               # Recent (2017-2021)
                         np.where(age < 15, 1.05,              # Modern (2010-2016)
                         np.where(age < 25, 0.95,              # Older (2000-2009)
                         np.where(age < 35, 0.82,              # Old (1990-1999)
                         0.70)))))                              # Very old (pre-1990)
        
        # Metro proximity - CRUCIAL in Bucharest
        metro_multiplier = np.where(metro_distance < 0.3, 1.20,        # <300m - Premium
                           np.where(metro_distance < 0.6, 1.12,        # 300-600m - Very good
                           np.where(metro_distance < 1.0, 1.05,        # 600m-1km - Good
                           np.where(metro_distance < 2.0, 0.98,        # 1-2km - Average
                           np.where(metro_distance < 3.5, 0.90,        # 2-3.5km - Below average
                           0.82)))))                                    # >3.5km - Far
        
        # Floor impact - realistic preferences
        floor_multiplier = np.where(floor == 0, 0.88,          # Ground floor - least desired
                           np.where(floor == 1, 0.94,          # First floor
                           np.where(floor <= 4, 1.04,          # Floors 2-4 - MOST desired
                           np.where(floor <= 7, 1.00,          # Floors 5-7 - Good
                           0.92))))                             # Floors 8+ - High floors
        
        # Number of rooms impact - MAJOR factor
        # This affects BOTH the price per sqm AND adds bonus
        rooms_multiplier = np.where(rooms == 1, 0.90,          # Studio - less desirable
                           np.where(rooms == 2, 1.00,          # 2 rooms - standard
                           np.where(rooms == 3, 1.12,          # 3 rooms - popular (+12%)
                           np.where(rooms == 4, 1.28,          # 4 rooms - large (+28%)
                           1.45))))                             # 5 rooms - very large (+45%)
        
        # Calculate BASE price from area and all multipliers
        base_price = (base_price_per_sqm * area * age_multiplier * 
                     metro_multiplier * floor_multiplier * rooms_multiplier)
        
        # BONUSES - Fixed amounts added
        bathrooms_bonus = np.where(bathrooms == 1, 0,
                          5000)                                 # 2nd bathroom adds 5000 EUR
        
        balcony_bonus = balcony * 2500                          # Balcony adds 2500 EUR
        
        # Additional room bonus - extra value per room beyond base
        extra_room_bonus = np.where(rooms <= 2, 0,
                           np.where(rooms == 3, 3000,           # 3rd room bonus
                           np.where(rooms == 4, 8000,           # 4th room larger bonus
                           15000)))                              # 5th room premium bonus
        
        # FINAL PRICE
        price = base_price + bathrooms_bonus + balcony_bonus + extra_room_bonus
        
        noise = np.random.normal(1, 0.08, n_samples)
        price = price * noise
        price = np.round(price, -2)
        
        data = pd.DataFrame({
            'Sector': sectors,
            'Suprafata_mp': np.round(area, 1),
            'Numar_Camere': rooms,
            'Etaj': floor,
            'An_Constructie': year_built,
            'Distanta_Metrou_km': np.round(metro_distance, 2),
            'Bai': bathrooms,
            'Balcon': balcony,
            'Pret_Euro': price
        })
        
        return data
    
    def calculate_market_statistics(self):
        """Calculează statistici detaliate despre piață"""
        if self.data is None:
            return
        
        # Statistics by sector
        for sector in range(1, 7):
            sector_data = self.data[self.data['Sector'] == sector]['Pret_Euro']
            self.sector_stats[sector] = {
                'mean': sector_data.mean(),
                'median': sector_data.median(),
                'std': sector_data.std(),
                'min': sector_data.min(),
                'max': sector_data.max(),
                'count': len(sector_data)
            }
        
        # Price per sqm analysis - calculate but don't add to dataframe permanently
        avg_price_per_sqm = (self.data['Pret_Euro'] / self.data['Suprafata_mp']).mean()
        
        # Market trends
        self.market_trends = {
            'avg_price': self.data['Pret_Euro'].mean(),
            'avg_price_per_sqm': avg_price_per_sqm,
            'total_properties': len(self.data),
            'price_range': (self.data['Pret_Euro'].min(), self.data['Pret_Euro'].max())
        }
    
    def save_data_and_model(self):
        """Salvează datele și modelul pe disc"""
        try:
            if self.data is not None:
                self.data.to_csv(self.data_path, index=False)
                print(f"✅ Date salvate în {self.data_path}")
            
            if self.model is not None:
                self.model.save(self.model_path)
                print(f"✅ Model salvat în {self.model_path}")
            
            if self.scaler is not None:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                print(f"✅ Scaler salvat în {self.scaler_path}")
            
            if self.feature_names is not None:
                with open('feature_names.json', 'w') as f:
                    json.dump(self.feature_names, f)
                print("✅ Feature names salvate")
                
            messagebox.showinfo("Succes", "Model și date salvate cu succes!")
            return True
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la salvare: {str(e)}")
            return False
    
    def load_saved_data_and_model(self):
        """Încarcă datele și modelul salvate"""
        try:
            if not (os.path.exists(self.model_path) and 
                   os.path.exists(self.data_path) and 
                   os.path.exists(self.scaler_path)):
                return False
            
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Date încărcate din {self.data_path}")
            
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model încărcat din {self.model_path}")
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler încărcat din {self.scaler_path}")
            
            if os.path.exists('feature_names.json'):
                with open('feature_names.json', 'r') as f:
                    self.feature_names = json.load(f)
            else:
                X = self.data.drop('Pret_Euro', axis=1)
                self.feature_names = X.columns.tolist()
            
            return True
        except Exception as e:
            print(f"❌ Eroare la încărcare: {str(e)}")
            return False
    
    def create_and_train_model(self, X_train, y_train, X_val, y_val):
        """Creează și antrenează modelul de rețea neuronală îmbunătățit"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            
            keras.layers.Dense(32, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        # Custom learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Enhanced callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return model, history
    
    def load_or_train_model(self):
        """Încarcă modelul existent sau antrenează unul nou"""
        if self.load_saved_data_and_model():
            print("✅ Model și date încărcate cu succes!")
            self.calculate_market_statistics()
            self.update_statistics_display()
            self.plot_all_charts()
        else:
            print("⚙️ Generare date și antrenare model nou...")
            self.data = self.generate_synthetic_data(n_samples=2000)
            
            X = self.data.drop('Pret_Euro', axis=1)
            y = self.data['Pret_Euro']
            self.feature_names = X.columns.tolist()
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model, self.history = self.create_and_train_model(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled, verbose=0)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n📊 Performanță Model:")
            print(f"   MSE: {mse:,.2f}")
            print(f"   MAE: {mae:,.2f}")
            print(f"   R² Score: {r2:.4f}")
            
            self.save_data_and_model()
            self.calculate_market_statistics()
            self.update_statistics_display()
            self.plot_all_charts()
    
    def create_ui(self):
        """Creează interfața utilizator îmbunătățită"""
        # Header
        header = tk.Frame(self.root, bg=self.colors['bg_medium'], height=80)
        header.pack(fill='x', padx=0, pady=0)
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header,
            text="🏠 UrbanPrice Bucharest",
            font=('Segoe UI', 28, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['accent']
        )
        title_label.pack(side='left', padx=30, pady=15)
        
        subtitle_label = tk.Label(
            header,
            text="Predicție Prețuri Imobiliare - București",
            font=('Segoe UI', 12),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack(side='left', padx=10, pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left Panel - Input & Prediction
        left_panel = tk.Frame(main_container, bg=self.colors['card'], width=450)
        left_panel.pack(side='left', fill='both', padx=(0, 10), pady=0)
        left_panel.pack_propagate(False)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Style for notebook - IMPROVED for better visibility
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=self.colors['card'], borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=self.colors['bg_light'],  # Light gray when not selected
                       foreground=self.colors['text'],       # Black text
                       padding=[15, 12],
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=2,
                       relief='solid')                       # Solid border
        style.map('TNotebook.Tab', 
                 background=[('selected', self.colors['accent'])],      # Red when selected
                 foreground=[('selected', 'white')],                    # White text when selected
                 relief=[('selected', 'raised')],
                 borderwidth=[('selected', 2)],
                 expand=[('selected', [1, 1, 1, 0])])
        
        self.create_input_tab()
        self.create_statistics_tab()
        self.create_history_tab()
        
        # Right Panel - Visualizations
        right_panel = tk.Frame(main_container, bg=self.colors['bg_dark'])
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Notebook for charts
        self.charts_notebook = ttk.Notebook(right_panel)
        self.charts_notebook.pack(fill='both', expand=True)
        
        self.create_visualization_tabs()
    
    def create_input_tab(self):
        """Tab pentru input și predicție"""
        tab1 = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(tab1, text='🎯 Predicție')
        
        # Scrollable frame with dark themed scrollbar
        canvas = tk.Canvas(tab1, bg=self.colors['card'], highlightthickness=0)
        
        # Custom dark scrollbar style
        style = ttk.Style()
        style.configure("Dark.Vertical.TScrollbar",
                       background=self.colors['bg_light'],
                       troughcolor=self.colors['bg_dark'],
                       bordercolor=self.colors['bg_dark'],
                       arrowcolor=self.colors['accent'],
                       relief='flat')
        style.map("Dark.Vertical.TScrollbar",
                 background=[('active', self.colors['accent']), 
                           ('pressed', self.colors['accent_hover'])])
        
        scrollbar = ttk.Scrollbar(tab1, orient="vertical", command=canvas.yview,
                                  style="Dark.Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas, bg=self.colors['card'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scroll support for input tab
        def _on_mousewheel_input(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_mousewheel_linux_input(event):
            if event.num == 4:
                canvas.yview_scroll(-3, "units")  # Scroll UP
            elif event.num == 5:
                canvas.yview_scroll(3, "units")   # Scroll DOWN
        
        def _bind_mousewheel_input(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel_input)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux_input)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux_input)
        
        def _unbind_mousewheel_input(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        
        canvas.bind("<Enter>", _bind_mousewheel_input)
        canvas.bind("<Leave>", _unbind_mousewheel_input)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Input fields
        tk.Label(
            scrollable_frame,
            text="📝 Caracteristici Proprietate",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['accent']
        ).pack(pady=(10, 20))
        
        self.create_modern_input(scrollable_frame, "Sector", "Sector", 1, 6, 1, "📍")
        self.create_modern_input(scrollable_frame, "Suprafață (mp)", "Suprafata_mp", 30, 200, 70, "📏")
        self.create_modern_input(scrollable_frame, "Număr Camere", "Numar_Camere", 1, 5, 2, "🚪")
        self.create_modern_input(scrollable_frame, "Etaj", "Etaj", 0, 10, 3, "🏢")
        self.create_modern_input(scrollable_frame, "An Construcție", "An_Constructie", 1970, 2024, 2010, "📅")
        self.create_modern_input(scrollable_frame, "Distanță Metrou (km)", "Distanta_Metrou_km", 0.1, 5, 1.0, "🚇")
        self.create_modern_input(scrollable_frame, "Băi", "Bai", 1, 2, 1, "🚿")
        self.create_modern_input(scrollable_frame, "Balcon (0/1)", "Balcon", 0, 1, 1, "🌿")
        
        # Prediction button
        predict_btn = tk.Button(
            scrollable_frame,
            text="🔮 CALCULEAZĂ PREȚ",
            command=self.predict_price,
            font=('Segoe UI', 13, 'bold'),
            bg=self.colors['accent'],
            fg=self.colors['bg_dark'],
            relief='flat',
            cursor='hand2',
            padx=30,
            pady=15,
            activebackground=self.colors['accent_hover']
        )
        predict_btn.pack(pady=20)
        
        # Result display
        result_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_dark'], relief='solid', bd=2)
        result_frame.pack(fill='x', padx=10, pady=10)
        
        self.result_label = tk.Label(
            result_frame,
            text="Apasă butonul pentru predicție",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary'],
            pady=30
        )
        self.result_label.pack()
        
        # Market comparison
        self.comparison_frame = tk.Frame(scrollable_frame, bg=self.colors['card'])
        self.comparison_frame.pack(fill='x', padx=10, pady=10)
        
        self.comparison_label = tk.Label(
            self.comparison_frame,
            text="",
            font=('Segoe UI', 10),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            justify='left'
        )
        self.comparison_label.pack()
        
        # Action buttons
        btn_frame = tk.Frame(scrollable_frame, bg=self.colors['card'])
        btn_frame.pack(pady=20)
        
        retrain_btn = tk.Button(
            btn_frame,
            text="🔄 Reantrenare Model",
            command=self.retrain_model,
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['warning'],
            fg=self.colors['bg_dark'],
            relief='flat',
            cursor='hand2',
            padx=15,
            pady=8
        )
        retrain_btn.pack(side='left', padx=5)
        
        save_btn = tk.Button(
            btn_frame,
            text="💾 Salvare Model",
            command=self.save_data_and_model,
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['success'],
            fg=self.colors['bg_dark'],
            relief='flat',
            cursor='hand2',
            padx=15,
            pady=8
        )
        save_btn.pack(side='left', padx=5)
    
    def create_statistics_tab(self):
        """Tab pentru statistici detaliate"""
        tab2 = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(tab2, text='📊 Statistici')
        
        # Fixed title at top
        tk.Label(
            tab2,
            text="📈 Statistici Piață Imobiliară",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['accent']
        ).pack(pady=(15, 10))
        
        # Scrollable text area
        stats_container = tk.Frame(tab2, bg=self.colors['card'])
        stats_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        self.stats_text = tk.Text(
            stats_container,
            font=('Consolas', 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            relief='flat',
            padx=20,
            pady=20,
            wrap='word'
        )
        self.stats_text.pack(side='left', fill='both', expand=True)
        
        # Dark themed scrollbar for stats
        stats_scroll = ttk.Scrollbar(stats_container, command=self.stats_text.yview,
                                     style="Dark.Vertical.TScrollbar")
        stats_scroll.pack(side='right', fill='y')
        self.stats_text.config(yscrollcommand=stats_scroll.set)
        
        # Mouse wheel scroll support for stats text
        def _on_mousewheel_stats(event):
            self.stats_text.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_mousewheel_linux_stats(event):
            if event.num == 4:
                self.stats_text.yview_scroll(-3, "units")
            elif event.num == 5:
                self.stats_text.yview_scroll(3, "units")
        
        self.stats_text.bind("<MouseWheel>", _on_mousewheel_stats)
        self.stats_text.bind("<Button-4>", _on_mousewheel_linux_stats)
        self.stats_text.bind("<Button-5>", _on_mousewheel_linux_stats)
    
    def create_history_tab(self):
        """Tab pentru istoric predicții"""
        tab3 = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(tab3, text='📜 Istoric')
        
        history_frame = tk.Frame(tab3, bg=self.colors['card'])
        history_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.history_text = tk.Text(
            history_frame,
            font=('Consolas', 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            insertbackground=self.colors['accent'],
            relief='flat',
            padx=15,
            pady=15
        )
        self.history_text.pack(side='left', fill='both', expand=True)
        
        history_scroll = ttk.Scrollbar(history_frame, command=self.history_text.yview,
                                       style="Dark.Vertical.TScrollbar")
        history_scroll.pack(side='right', fill='y')
        self.history_text.config(yscrollcommand=history_scroll.set)
        
        # Mouse wheel scroll support for history text
        def _on_mousewheel_history(event):
            self.history_text.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_mousewheel_linux_history(event):
            if event.num == 4:
                self.history_text.yview_scroll(-1, "units")
            elif event.num == 5:
                self.history_text.yview_scroll(1, "units")
        
        self.history_text.bind("<MouseWheel>", _on_mousewheel_history)
        self.history_text.bind("<Button-4>", _on_mousewheel_linux_history)
        self.history_text.bind("<Button-5>", _on_mousewheel_linux_history)
        
        btn_frame = tk.Frame(tab3, bg=self.colors['card'])
        btn_frame.pack(pady=10)
        
        clear_btn = tk.Button(
            btn_frame,
            text="🗑️ Șterge Istoric",
            command=self.clear_history,
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['danger'],
            fg=self.colors['text'],
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10
        )
        clear_btn.pack(side='left', padx=5)
        
        export_btn = tk.Button(
            btn_frame,
            text="📄 Export CSV",
            command=self.export_history,
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['blue'],
            fg=self.colors['text'],
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10
        )
        export_btn.pack(side='left', padx=5)
    
    def create_visualization_tabs(self):
        """Creează tab-uri pentru diverse grafice"""
        # Tab 1: Training History
        tab1 = tk.Frame(self.charts_notebook, bg=self.colors['bg_dark'])
        self.charts_notebook.add(tab1, text='📈 Antrenare Model')
        
        self.fig1 = Figure(figsize=(10, 6), facecolor=self.colors['bg_dark'])
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor(self.colors['bg_medium'])
        self.canvas1 = FigureCanvasTkAgg(self.fig1, tab1)
        self.canvas1.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 2: Price Distribution by Sector
        tab2 = tk.Frame(self.charts_notebook, bg=self.colors['bg_dark'])
        self.charts_notebook.add(tab2, text='🏘️ Distribuție Prețuri')
        
        self.fig2 = Figure(figsize=(10, 6), facecolor=self.colors['bg_dark'])
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor(self.colors['bg_medium'])
        self.canvas2 = FigureCanvasTkAgg(self.fig2, tab2)
        self.canvas2.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 3: Correlation Heatmap
        tab3 = tk.Frame(self.charts_notebook, bg=self.colors['bg_dark'])
        self.charts_notebook.add(tab3, text='🔥 Heatmap ')
        
        self.fig3 = Figure(figsize=(10, 6), facecolor=self.colors['bg_dark'])
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_facecolor(self.colors['bg_medium'])
        self.canvas3 = FigureCanvasTkAgg(self.fig3, tab3)
        self.canvas3.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 4: Price vs Area Scatter
        tab4 = tk.Frame(self.charts_notebook, bg=self.colors['bg_dark'])
        self.charts_notebook.add(tab4, text='📊 Preț vs Suprafață')
        
        self.fig4 = Figure(figsize=(10, 6), facecolor=self.colors['bg_dark'])
        self.ax4 = self.fig4.add_subplot(111)
        self.ax4.set_facecolor(self.colors['bg_medium'])
        self.canvas4 = FigureCanvasTkAgg(self.fig4, tab4)
        self.canvas4.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 5: Feature Importance
        tab5 = tk.Frame(self.charts_notebook, bg=self.colors['bg_dark'])
        self.charts_notebook.add(tab5, text='⚡ Importanță Dotări')
        
        self.fig5 = Figure(figsize=(10, 6), facecolor=self.colors['bg_dark'])
        self.ax5 = self.fig5.add_subplot(111)
        self.ax5.set_facecolor(self.colors['bg_medium'])
        self.canvas5 = FigureCanvasTkAgg(self.fig5, tab5)
        self.canvas5.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 6: Predictions vs Actual
        tab6 = tk.Frame(self.charts_notebook, bg=self.colors['bg_dark'])
        self.charts_notebook.add(tab6, text='🎯 Predicții vs Realitate')
        
        self.fig6 = Figure(figsize=(10, 6), facecolor=self.colors['bg_dark'])
        self.ax6 = self.fig6.add_subplot(111)
        self.ax6.set_facecolor(self.colors['bg_medium'])
        self.canvas6 = FigureCanvasTkAgg(self.fig6, tab6)
        self.canvas6.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_modern_input(self, parent, label_text, var_name, min_val, max_val, default, icon):
        """Creează un input modern cu design îmbunătățit"""
        container = tk.Frame(parent, bg=self.colors['card'])
        container.pack(fill='x', pady=12)
        
        label_frame = tk.Frame(container, bg=self.colors['card'])
        label_frame.pack(fill='x')
        
        tk.Label(
            label_frame,
            text=f"{icon} {label_text}",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text'],
            anchor='w'
        ).pack(side='left')
        
        var = tk.DoubleVar(value=default)
        self.inputs[var_name] = var
        
        value_label = tk.Label(
            label_frame,
            textvariable=var,
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['accent'],
            width=10
        )
        value_label.pack(side='right')
        
        step = 0.1 if var_name == 'Distanta_Metrou_km' else 1
        
        slider = tk.Scale(
            container,
            from_=min_val,
            to=max_val,
            resolution=step,
            orient='horizontal',
            variable=var,
            showvalue=False,
            bg=self.colors['card'],
            fg=self.colors['accent'],              # Red color for slider
            troughcolor=self.colors['bg_light'],    # Light gray trough
            activebackground=self.colors['accent_hover'],  # Dark red when dragging
            highlightthickness=0,
            sliderrelief='raised',                  # Raised relief for better visibility
            borderwidth=2,                          # Border for slider button
            length=400
        )
        slider.pack(fill='x', pady=(5, 0))
    
    def predict_price(self):
        """Efectuează predicția prețului cu analiză detaliată"""
        if self.model is None:
            messagebox.showerror("Eroare", "Modelul nu este antrenat!")
            return
        
        try:
            input_data = []
            for feature in self.feature_names:
                value = self.inputs[feature].get()
                input_data.append(value)
            
            input_df = pd.DataFrame([input_data], columns=self.feature_names)
            input_scaled = self.scaler.transform(input_df)
            prediction = self.model.predict(input_scaled, verbose=0)[0][0]
            
            # Get sector info
            sector = int(input_data[0])
            area = input_data[1]
            price_per_sqm = prediction / area
            
            # Compare with market
            sector_avg = self.sector_stats[sector]['mean']
            market_avg = self.market_trends['avg_price']
            diff_sector = ((prediction - sector_avg) / sector_avg) * 100
            diff_market = ((prediction - market_avg) / market_avg) * 100
            
            # Display result
            result_text = f"💰 Preț Estimat\n{prediction:,.0f} €\n{prediction * 5:,.0f} RON\n\n"
            result_text += f"📏 Preț/mp: {price_per_sqm:,.0f} €/mp"
            
            self.result_label.config(
                text=result_text,
                fg=self.colors['success']
            )
            
            # Comparison text
            comparison_text = f"📊 Comparație cu Piața:\n\n"
            comparison_text += f"Sector {sector} (Media: {sector_avg:,.0f} €):\n"
            comparison_text += f"{'🔺' if diff_sector > 0 else '🔻'} {abs(diff_sector):.1f}% {'peste' if diff_sector > 0 else 'sub'} medie\n\n"
            comparison_text += f"Piață Generală (Media: {market_avg:,.0f} €):\n"
            comparison_text += f"{'🔺' if diff_market > 0 else '🔻'} {abs(diff_market):.1f}% {'peste' if diff_market > 0 else 'sub'} medie"
            
            self.comparison_label.config(text=comparison_text)
            
            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = (f"[{timestamp}]\n"
                           f"Preț: {prediction:,.0f} € ({prediction * 5:,.0f} RON)\n"
                           f"Sector: {sector} | Suprafață: {area:.0f}mp | Preț/mp: {price_per_sqm:,.0f} €\n"
                           f"Camere: {int(input_data[2])} | Etaj: {int(input_data[3])} | "
                           f"An: {int(input_data[4])}\n"
                           f"{'-'*60}\n")
            
            self.prediction_history.append({
                'timestamp': timestamp,
                'prediction': prediction,
                'features': input_data
            })
            self.history_text.insert('1.0', history_entry)
            
            # Update current prediction chart
            self.plot_current_prediction(input_data, prediction)
            
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la predicție: {str(e)}")
    
    def update_statistics_display(self):
        """Actualizează afișarea statisticilor"""
        if not self.sector_stats:
            return
        
        self.stats_text.delete('1.0', 'end')
        
        stats_content = "═══════════════════════════════════════════════════════\n"
        stats_content += "                 STATISTICI PIAȚĂ BUCUREȘTI\n"
        stats_content += "═══════════════════════════════════════════════════════\n\n"
        
        stats_content += f"📊 OVERVIEW GENERAL\n"
        stats_content += f"{'─'*55}\n"
        stats_content += f"Total Proprietăți Analizate: {self.market_trends['total_properties']:,}\n"
        stats_content += f"Preț Mediu Piață: {self.market_trends['avg_price']:,.0f} €\n"
        stats_content += f"Preț Mediu/mp: {self.market_trends['avg_price_per_sqm']:,.0f} €/mp\n"
        stats_content += f"Interval Prețuri: {self.market_trends['price_range'][0]:,.0f} - "
        stats_content += f"{self.market_trends['price_range'][1]:,.0f} €\n\n"
        
        stats_content += f"🏘️ STATISTICI PE SECTOARE\n"
        stats_content += f"{'═'*55}\n\n"
        
        for sector in range(1, 7):
            stats = self.sector_stats[sector]
            stats_content += f"SECTOR {sector}\n"
            stats_content += f"{'─'*55}\n"
            stats_content += f"  Număr Proprietăți: {stats['count']}\n"
            stats_content += f"  Preț Mediu: {stats['mean']:,.0f} €\n"
            stats_content += f"  Preț Median: {stats['median']:,.0f} €\n"
            stats_content += f"  Deviație Standard: {stats['std']:,.0f} €\n"
            stats_content += f"  Minim: {stats['min']:,.0f} €\n"
            stats_content += f"  Maxim: {stats['max']:,.0f} €\n"
            stats_content += f"\n"
        
        self.stats_text.insert('1.0', stats_content)
    
    def plot_all_charts(self):
        """Generează toate graficele"""
        if self.data is None:
            return
        
        self.plot_training_history()
        self.plot_price_distribution()
        self.plot_correlation_heatmap()
        self.plot_price_vs_area()
        self.plot_feature_importance()
        self.plot_predictions_vs_actual()
    
    def plot_training_history(self):
        """Grafic istoric antrenare"""
        self.ax1.clear()
        
        if self.history is None:
            # Show message if no training history available
            self.ax1.text(0.5, 0.5, 
                         'Modelul a fost încărcat din fișier.\n\n'
                         'Pentru a vedea graficul de antrenare,\n'
                         'apasă butonul "🔄 Reantrenare Model".',
                         transform=self.ax1.transAxes,
                         ha='center', va='center',
                         fontsize=14, color=self.colors['text'],
                         bbox=dict(boxstyle='round', facecolor=self.colors['card'], 
                                 alpha=0.8, edgecolor=self.colors['accent'], linewidth=2))
            self.ax1.set_facecolor(self.colors['bg_medium'])
            self.ax1.tick_params(colors=self.colors['text'])
            self.fig1.tight_layout()
            self.canvas1.draw()
            return
        
        epochs = range(1, len(self.history.history['loss']) + 1)
        
        self.ax1.plot(epochs, self.history.history['loss'], 
                     color=self.colors['accent'], linewidth=2.5, 
                     label='Loss Antrenare', marker='o', markersize=5)
        self.ax1.plot(epochs, self.history.history['val_loss'], 
                     color=self.colors['blue'], linewidth=2.5, 
                     label='Loss Validare', marker='s', markersize=5)
        
        self.ax1.set_xlabel('Epoci', fontsize=12, color=self.colors['text'], fontweight='bold')
        self.ax1.set_ylabel('Loss (MSE)', fontsize=12, color=self.colors['text'], fontweight='bold')
        self.ax1.set_title('Evoluția Antrenării Modelului Neural Network', 
                          fontsize=14, fontweight='bold', color=self.colors['text'], pad=20)
        self.ax1.legend(facecolor='white', edgecolor=self.colors['accent'], 
                       fontsize=10, loc='best')
        for text in self.ax1.legend().get_texts():
          text.set_color(self.colors['text'])
        self.ax1.grid(True, alpha=0.3, color=self.colors['text_secondary'], linestyle='-', linewidth=0.8)
        self.ax1.tick_params(colors=self.colors['text'], labelsize=10)
        
        self.fig1.tight_layout()
        self.canvas1.draw()
    
    def plot_price_distribution(self):
        """Grafic distribuție prețuri pe sectoare - IMPROVED VISIBILITY"""
        self.ax2.clear()
        
        sector_data = [self.data[self.data['Sector'] == i]['Pret_Euro'].values 
                      for i in range(1, 7)]
        
        # Red-centered professional color palette
        colors_list = [self.colors['red_dark'], self.colors['accent'], 
                      self.colors['pink'], self.colors['orange'], 
                      self.colors['warning'], self.colors['success']]
        
        bp = self.ax2.boxplot(sector_data, patch_artist=True, 
                              labels=[f'Sector {i}' for i in range(1, 7)],
                              showmeans=True, meanline=True,
                              widths=0.6)
        
        # Make boxes MORE VISIBLE on white background
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(2)
            patch.set_edgecolor(self.colors['text'])
        
        # Make whiskers, caps, means, medians MORE VISIBLE
        for element in ['whiskers', 'caps']:
            plt.setp(bp[element], color=self.colors['text'], linewidth=2)
        plt.setp(bp['means'], color=self.colors['text'], linewidth=3)
        plt.setp(bp['medians'], color=self.colors['accent'], linewidth=3.5)
        plt.setp(bp['fliers'], marker='o', markerfacecolor=self.colors['accent'], 
                markersize=8, markeredgecolor=self.colors['text'], linewidth=1.5)
        
        self.ax2.set_xlabel('Sector', fontsize=13, color=self.colors['text'], fontweight='bold')
        self.ax2.set_ylabel('Preț (€)', fontsize=13, color=self.colors['text'], fontweight='bold')
        self.ax2.set_title('Distribuția Prețurilor pe Sectoare București', 
                          fontsize=15, fontweight='bold', color=self.colors['text'], pad=20)
        self.ax2.grid(True, alpha=0.3, axis='y', color=self.colors['text_secondary'], 
                     linestyle='-', linewidth=0.8)
        self.ax2.tick_params(colors=self.colors['text'], labelsize=11)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['red_dark'], edgecolor=self.colors['text'], label='Sector 1'),
            Patch(facecolor=self.colors['accent'], edgecolor=self.colors['text'], label='Sector 2'),
            Patch(facecolor=self.colors['pink'], edgecolor=self.colors['text'], label='Sector 3'),
            Patch(facecolor=self.colors['orange'], edgecolor=self.colors['text'], label='Sector 4'),
            Patch(facecolor=self.colors['warning'], edgecolor=self.colors['text'], label='Sector 5'),
            Patch(facecolor=self.colors['success'], edgecolor=self.colors['text'], label='Sector 6')
            ]
        self.ax2.legend(handles=legend_elements, 
               facecolor='white', 
               edgecolor=self.colors['accent'],
               fontsize=11, 
               loc='upper right',
               framealpha=0.95,
               labelcolor=self.colors['text'])
        
        
        self.fig2.tight_layout()
        self.canvas2.draw()
    
    def plot_correlation_heatmap(self):
        """Heatmap corelație între features"""
        # Clear the entire figure (removes colorbar too)
        self.fig3.clf()
        
        # Recreate the axis
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_facecolor(self.colors['bg_medium'])
        
        corr_matrix = self.data.corr()
        
        im = self.ax3.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', 
                            vmin=-1, vmax=1, interpolation='nearest')
        
        self.ax3.set_xticks(np.arange(len(corr_matrix.columns)))
        self.ax3.set_yticks(np.arange(len(corr_matrix.columns)))
        self.ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', 
                                fontsize=9, color=self.colors['text'])
        self.ax3.set_yticklabels(corr_matrix.columns, fontsize=9, 
                                color=self.colors['text'])
        
        # Add correlation values with better contrast
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = self.ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", 
                                   color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else self.colors['text'],
                                   fontsize=9, fontweight='bold')
        
        self.ax3.set_title('Matricea de Corelație - Dotări vs Preț', 
                          fontsize=14, fontweight='bold', color=self.colors['text'], pad=20)
        
        # Colorbar - only one now since we cleared the figure
        cbar = self.fig3.colorbar(im, ax=self.ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Coeficient Corelație', rotation=270, labelpad=20, 
                      color=self.colors['text'], fontweight='bold')
        cbar.ax.tick_params(colors=self.colors['text'])
        
        self.fig3.tight_layout()
        self.canvas3.draw()
    
    def plot_price_vs_area(self):
        """Scatter plot preț vs suprafață - IMPROVED VISIBILITY"""
        self.ax4.clear()
        
        sectors = self.data['Sector'].unique()
        colors_map = {1: self.colors['red_dark'], 2: self.colors['accent'], 
                     3: self.colors['pink'], 4: self.colors['orange'],
                     5: self.colors['warning'], 6: self.colors['success']}
        
        for sector in sorted(sectors):
            sector_data = self.data[self.data['Sector'] == sector]
            self.ax4.scatter(sector_data['Suprafata_mp'], sector_data['Pret_Euro'],
                           alpha=0.6, s=70,
                           c=colors_map[sector], 
                           label=f'Sector {sector}', 
                           edgecolors=self.colors['text'], linewidth=1)
        
        # Trend line MORE VISIBLE
        z = np.polyfit(self.data['Suprafata_mp'], self.data['Pret_Euro'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.data['Suprafata_mp'].min(), 
                            self.data['Suprafata_mp'].max(), 100)
        self.ax4.plot(x_line, p(x_line), color=self.colors['accent'], 
                     linewidth=3.5, linestyle='--', label='Trend General', alpha=0.9)
        
        self.ax4.set_xlabel('Suprafață (mp)', fontsize=13, color=self.colors['text'], 
                           fontweight='bold')
        self.ax4.set_ylabel('Preț (€)', fontsize=13, color=self.colors['text'], 
                           fontweight='bold')
        self.ax4.set_title('Relația Preț - Suprafață pe Sectoare', 
                          fontsize=15, fontweight='bold', color=self.colors['text'], pad=20)
        
        # Legend for white background
        legend = self.ax4.legend(facecolor='white', 
                                edgecolor=self.colors['accent'], 
                                fontsize=11, loc='upper left', 
                                framealpha=0.95, ncol=2)
        for text in legend.get_texts():
          text.set_color(self.colors['text'])
        legend.get_frame().set_linewidth(2)
        
        self.ax4.grid(True, alpha=0.3, color=self.colors['text_secondary'], 
                     linestyle='-', linewidth=0.8)
        self.ax4.tick_params(colors=self.colors['text'], labelsize=11)
        
        self.fig4.tight_layout()
        self.canvas4.draw()
    
    def plot_feature_importance(self):
        """Grafic importanță features (bazat pe corelație)"""
        self.ax5.clear()
        
        correlations = self.data.corr()['Pret_Euro'].drop('Pret_Euro').abs().sort_values(ascending=True)
        
        colors_bars = [self.colors['accent'] if x > correlations.median() 
                      else self.colors['blue'] for x in correlations]
        
        bars = self.ax5.barh(range(len(correlations)), correlations.values, 
                            color=colors_bars, edgecolor=self.colors['text'], linewidth=1.5)
        
        self.ax5.set_yticks(range(len(correlations)))
        self.ax5.set_yticklabels(correlations.index, fontsize=11, color=self.colors['text'])
        
        for i, (bar, value) in enumerate(zip(bars, correlations.values)):
            self.ax5.text(value + 0.02, i, f'{value:.3f}',
                         va='center', fontsize=10, fontweight='bold',
                         color=self.colors['text'])
        
        self.ax5.set_xlabel('Coeficient Corelație (absolute)', fontsize=12, 
                           color=self.colors['text'], fontweight='bold')
        self.ax5.set_title('Importanța Dotărilor în Determinarea Prețului', 
                          fontsize=14, fontweight='bold', color=self.colors['text'], pad=20)
        self.ax5.grid(True, alpha=0.3, axis='x', color=self.colors['text_secondary'], 
                     linestyle='--')
        self.ax5.tick_params(colors=self.colors['text'], labelsize=10)
        
        self.fig5.tight_layout()
        self.canvas5.draw()
    
    def plot_predictions_vs_actual(self):
        """Grafic predicții vs valori reale - IMPROVED VISIBILITY"""
        self.ax6.clear()
        
        try:
            # Make predictions on entire dataset - exclude Pret_Euro only
            X = self.data[self.feature_names]  # Use only the original feature names
            y_true = self.data['Pret_Euro']
            X_scaled = self.scaler.transform(X)
            y_pred = self.model.predict(X_scaled, verbose=0).flatten()
            
            # Calculate R²
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Scatter plot - on white background
            self.ax6.scatter(y_true, y_pred, alpha=0.5, s=50, 
                            c=self.colors['accent'], edgecolors=self.colors['text'], linewidth=0.8)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            self.ax6.plot([min_val, max_val], [min_val, max_val], 
                         color=self.colors['text'], linewidth=3, 
                         linestyle='--', label='Predicție Perfectă', alpha=0.8)
            
            self.ax6.set_xlabel('Preț Real (€)', fontsize=13, color=self.colors['text'], 
                               fontweight='bold')
            self.ax6.set_ylabel('Preț Prezis (€)', fontsize=13, color=self.colors['text'], 
                               fontweight='bold')
            self.ax6.set_title(f'Acuratețea Modelului (R² = {r2:.4f})', 
                              fontsize=15, fontweight='bold', color=self.colors['text'], pad=20)
            
            # Legend for white background
            legend = self.ax6.legend(facecolor='white', 
                                    edgecolor=self.colors['accent'], 
                                    fontsize=11, loc='best', framealpha=0.95)
            for text in legend.get_texts():
              text.set_color(self.colors['text'])
            legend.get_frame().set_linewidth(2)
            
            self.ax6.grid(True, alpha=0.3, color=self.colors['text_secondary'], 
                         linestyle='-', linewidth=0.8)
            self.ax6.tick_params(colors=self.colors['text'], labelsize=11)
            
            # Add R² text box
            textstr = f'R² Score: {r2:.4f}\nMAE: {mae:,.0f} €'
            props = dict(boxstyle='round', facecolor='white', 
                        alpha=0.95, edgecolor=self.colors['accent'], linewidth=2)
            self.ax6.text(0.05, 0.95, textstr, transform=self.ax6.transAxes, 
                         fontsize=12, verticalalignment='top', bbox=props,
                         color=self.colors['text'], fontweight='bold')
            
            self.fig6.tight_layout()
            self.canvas6.draw()
            
            print(f"✅ Grafic Predicții vs Realitate generat (R²: {r2:.4f}, MAE: {mae:,.0f}€)")
            
        except Exception as e:
            print(f"❌ Eroare la generare grafic Predicții vs Realitate: {str(e)}")
            # Show error message on plot
            self.ax6.text(0.5, 0.5, f'Eroare la generare grafic:\n{str(e)}', 
                         transform=self.ax6.transAxes, ha='center', va='center',
                         fontsize=12, color=self.colors['danger'], 
                         bbox=dict(boxstyle='round', facecolor=self.colors['bg_light'], alpha=0.8))
            self.fig6.tight_layout()
            self.canvas6.draw()
    
    def plot_current_prediction(self, input_data, prediction):
        """Update pentru predicția curentă (poate fi adăugat un nou tab dacă dorești)"""
        # This could be expanded to show current prediction analysis
        pass
    
    def retrain_model(self):
        """Reantrenează modelul cu date noi"""
        response = messagebox.askyesno(
            "Reantrenare Model",
            "Sigur vrei să reantrenezi modelul?\n\n"
            "Acest proces va:\n"
            "• Genera date noi\n"
            "• Antrena un model nou\n"
            "• Dura 30-60 secunde\n\n"
            "Continuăm?"
        )
        
        if response:
            try:
                # Delete old files
                for file in [self.model_path, self.data_path, self.scaler_path]:
                    if os.path.exists(file):
                        if os.path.isdir(file):
                            import shutil
                            shutil.rmtree(file)
                        else:
                            os.remove(file)
                
                # Retrain
                self.model = None
                self.scaler = None
                self.history = None
                self.load_or_train_model()
                
                messagebox.showinfo("Succes", "Model reantrenat cu succes!")
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la reantrenare: {str(e)}")
    
    def clear_history(self):
        """Șterge istoricul predicțiilor"""
        self.prediction_history = []
        self.history_text.delete('1.0', 'end')
        messagebox.showinfo("Succes", "Istoric șters!")
    
    def export_history(self):
        """Exportă istoricul în CSV"""
        if not self.prediction_history:
            messagebox.showwarning("Atenție", "Nu există predicții în istoric!")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"istoric_predictii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if filename:
                df_history = pd.DataFrame([
                    {
                        'Timestamp': entry['timestamp'],
                        'Pret_Prezis_EUR': entry['prediction'],
                        'Pret_Prezis_RON': entry['prediction'] * 5,
                        **{name: val for name, val in zip(self.feature_names, entry['features'])}
                    }
                    for entry in self.prediction_history
                ])
                
                df_history.to_csv(filename, index=False)
                messagebox.showinfo("Succes", f"Istoric exportat în:\n{filename}")
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la export: {str(e)}")

def main():
    root = tk.Tk()
    app = HousePricePredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()