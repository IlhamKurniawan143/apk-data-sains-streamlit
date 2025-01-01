import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('dataset.csv')  # Sesuaikan dengan path dataset Anda

# RBF Neural Network implementation
class RBFNN:
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.mlp = MLPClassifier(hidden_layer_sizes=(), activation='identity', max_iter=500, random_state=random_state)

    def fit(self, X, y):
        self.kmeans.fit(X)
        rbf_features = self._transform_rbf(X)
        self.mlp.fit(rbf_features, y)

    def predict(self, X):
        rbf_features = self._transform_rbf(X)
        return self.mlp.predict(rbf_features)

    def _transform_rbf(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.kmeans.cluster_centers_, axis=2)
        return np.exp(-distances**2)

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
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Standarisasi data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Inisialisasi model
    if algorithm == "Naive Bayes":
        model = GaussianNB()
    elif algorithm == "RBF Neural Network":
        model = RBFNN(n_clusters=10, random_state=42)

    model.fit(X_train, y_train)

    if page == "Training dan Validasi":
        st.header("Halaman Training dan Validasi")

        # Reset indeks sebelum menggabungkan
        X_train_reset = pd.DataFrame(scaler.inverse_transform(X_train), columns=data.columns[:-1])
        y_train_reset = y_train.reset_index(drop=True)
        training_data = pd.concat([X_train_reset, y_train_reset], axis=1)

        # Tampilkan data training lengkap
        st.write("Data Training:")
        st.write(training_data)

        # Hasil Training
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')

        st.subheader("Hasil Training")
        st.write(f"Akurasi Training: {train_accuracy:.2f}")
        st.write(f"Precision Training: {train_precision:.2f}")
        st.write(f"Recall Training: {train_recall:.2f}")
        st.write(f"F1 Score Training: {train_f1:.2f}")

        # Hasil Validasi
        y_val_pred = model.predict(X_val)
        validation_accuracy = accuracy_score(y_val, y_val_pred)
        validation_precision = precision_score(y_val, y_val_pred, average='weighted')
        validation_recall = recall_score(y_val, y_val_pred, average='weighted')
        validation_f1 = f1_score(y_val, y_val_pred, average='weighted')

        st.subheader("Hasil Validasi")
        st.write(f"Akurasi Validasi: {validation_accuracy:.2f}")
        st.write(f"Precision Validasi: {validation_precision:.2f}")
        st.write(f"Recall Validasi: {validation_recall:.2f}")
        st.write(f"F1 Score Validasi: {validation_f1:.2f}")

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
            else:
                # Input default jika ada kolom lain
                inputs[col] = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0, step=0.01)

        if st.button("Klasifikasi Kelulusan"):
            input_values = np.array([list(inputs.values())])
            input_scaled = scaler.transform(input_values)
            prediction = model.predict(input_scaled)
            st.write(f"Hasil Klasifikasi: {prediction[0]}")

        # Upload file untuk testing
        st.subheader("Upload File CSV untuk Testing")
        uploaded_test_file = st.file_uploader("Pilih File CSV", type="csv")
        if uploaded_test_file:
            test_data = pd.read_csv(uploaded_test_file)
            X_new_test = test_data.iloc[:, :-1]
            y_new_test = test_data.iloc[:, -1]
            X_new_test_scaled = scaler.transform(X_new_test)
            y_new_pred = model.predict(X_new_test_scaled)

            # Tampilkan hasil prediksi
            test_data["Prediction"] = y_new_pred
            st.write("Hasil Prediksi untuk Data Testing:")
            st.write(test_data)

            # Evaluasi
            test_accuracy = accuracy_score(y_new_test, y_new_pred)
            st.write(f"Akurasi Testing: {test_accuracy:.2f}")
