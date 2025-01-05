import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Tempatkan class RBFNN di sini setelah semua import
class RBFNN:
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=2000, solver='adam', random_state=random_state)
        self.spread = None

    def fit(self, X, y):
        self.kmeans.fit(X)
        distances = np.linalg.norm(
            self.kmeans.cluster_centers_[:, np.newaxis] - self.kmeans.cluster_centers_, axis=2
        )
        self.spread = np.mean(distances) / np.sqrt(2 * self.n_clusters)
        rbf_features = self._transform_rbf(X)
        self.mlp.fit(rbf_features, y)

    def predict(self, X):
        rbf_features = self._transform_rbf(X)
        return self.mlp.predict(rbf_features)

    def _transform_rbf(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.kmeans.cluster_centers_, axis=2)
        return np.exp(-(distances ** 2) / (2 * (self.spread ** 2)))

# Naive Bayes Manual Implementation
class ManualNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(X)
            self.feature_stats[cls] = {
                "mean": X_cls.mean(axis=0),
                "var": X_cls.var(axis=0)
            }

    def _gaussian_likelihood(self, x, mean, var):
        eps = 1e-6
        coeff = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coeff * exponent

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for cls in self.classes:
                prior = self.class_priors[cls]
                likelihood = np.prod(self._gaussian_likelihood(x, self.feature_stats[cls]["mean"], self.feature_stats[cls]["var"]))
                posteriors[cls] = prior * likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('dataset.csv')  # Sesuaikan dengan path dataset Anda

# Load dataset
data = load_data()

# Page configurations
st.title("Aplikasi Klasifikasi Jenis Kelulusan Mahasiswa")
st.sidebar.title("Navigasi")

# Sidebar navigation
page = st.sidebar.radio("Pilih Halaman:", ["Dataset", "Training dan Validasi", "Testing"])

# Pilihan algoritma
st.sidebar.subheader("Pilih Algoritma")
algorithm = st.sidebar.radio("Algoritma", ["Naive Bayes", "RBF Neural Network"])

if page == "Dataset":
    st.header("Halaman Dataset")
    st.write("Dataset:")
    st.write(data)

else:
    # Split dataset sesuai alokasi data
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Standarisasi data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Inisialisasi model
    if algorithm == "Naive Bayes":
        model = ManualNaiveBayes()
    elif algorithm == "RBF Neural Network":
        model = RBFNN(n_clusters=10, random_state=42)

    model.fit(X_train, y_train)

    if page == "Training dan Validasi":
        st.header("Halaman Training dan Validasi")
        st.write(data)

        # Hasil Training
        y_train_pred = model.predict(X_train)
        train_report = classification_report(y_train, y_train_pred, output_dict=True)

        st.subheader("Hasil Training")
        st.write(f"Akurasi Training: {train_report['accuracy']:.2f}")
        st.write(f"Precision Training: {train_report['weighted avg']['precision']:.2f}")
        st.write(f"Recall Training: {train_report['weighted avg']['recall']:.2f}")
        st.write(f"F1 Score Training: {train_report['weighted avg']['f1-score']:.2f}")

        # Hasil Validasi
        y_val_pred = model.predict(X_val)
        val_report = classification_report(y_val, y_val_pred, output_dict=True)

        st.subheader("Hasil Validasi")
        st.write(f"Akurasi Validasi: {val_report['accuracy']:.2f}")
        st.write(f"Precision Validasi: {val_report['weighted avg']['precision']:.2f}")
        st.write(f"Recall Validasi: {val_report['weighted avg']['recall']:.2f}")
        st.write(f"F1 Score Validasi: {val_report['weighted avg']['f1-score']:.2f}")

    elif page == "Testing":
        st.header("Halaman Testing")

        # Input manual untuk testing
        st.subheader("Input Variabel untuk Pengujian Manual")
        inputs = {}
        for col in data.columns[:-1]:
            if col in ["CUTI", "MATA KULIAH TIDAK LULUS", "UMUR"]:
                # Input angka tanpa desimal
                inputs[col] = st.number_input(f"Masukkan nilai untuk {col} (tanpa desimal):", value=0, step=1, format="%d")
            elif col in ["IPK", "IPS1", "IPS2", "IPS3", "IPS4"]:
                # Input angka dengan desimal
                inputs[col] = st.number_input(f"Masukkan nilai untuk {col} (dengan desimal):", value=0.0, step=0.01, format="%.2f")

        if st.button("Klasifikasi Kelulusan"):
            input_values = np.array([list(inputs.values())])
            input_scaled = scaler.transform(input_values)
            prediction = model.predict(input_scaled)
            st.write(f"Hasil Klasifikasi: {prediction[0]}")

        # Upload file untuk testing
        st.subheader("Upload File CSV untuk Testing")
        uploaded_test_file = st.file_uploader("Pilih File CSV", type="csv")
        if uploaded_test_file:
            try:
                test_data = pd.read_csv(uploaded_test_file)

                # Pastikan kolom dataset baru cocok dengan kolom dataset saat fit
                missing_features = set(data.columns[:-1]) - set(test_data.columns)
                extra_features = set(test_data.columns) - set(data.columns[:-1])

                if missing_features:
                    st.error(f"Dataset yang diunggah tidak memiliki kolom: {', '.join(missing_features)}")
                elif extra_features:
                    st.error(f"Dataset yang diunggah memiliki kolom tambahan: {', '.join(extra_features)}")
                else:
                    # Transformasi dan prediksi jika kolom sesuai
                    X_new_test = test_data[data.columns[:-1]].values
                    X_new_test_scaled = scaler.transform(X_new_test)
                    y_new_pred = model.predict(X_new_test_scaled)

                    # Tambahkan hasil prediksi ke dataset
                    test_data["Klasifikasi"] = y_new_pred
                    st.write("Hasil Klasifikasi untuk Data Testing:")
                    st.write(test_data)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
        else:
            st.warning("Silakan unggah file CSV untuk melakukan pengujian.")
