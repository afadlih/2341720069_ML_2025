# Machine Learning - Semester 5
### Jobsheet Machine Learning

**Nama:** Ahmad Fadlih Wahyu Sardana  
**NIM:** 2341720069  
**Program Studi:** D4 Teknik Informatika  
**Semester:** 5 (2025)

**Sumber Materi:** [JTI Modul Praktikum Pembelajaran Mesin](https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-mah/)

---

## Deskripsi

Repository ini berisi kumpulan project dan praktikum Machine Learning yang dikerjakan selama semester 5. Setiap jobsheet mencakup implementasi berbagai algoritma machine learning, mulai dari dasar-dasar Python untuk data science hingga teknik-teknik advanced seperti deep learning dan natural language processing.

Semua materi praktikum mengacu pada modul resmi dari Jurusan Teknologi Informasi (JTI) Politeknik Negeri Malang yang tersedia di GitBook.

---

## Struktur Repository

```
2341720069_ML_2025/
│
├── JS01/  - Perkenalan Machine Learning & Python Basics
├── JS02/  - Visualisasi Data & Matplotlib
├── JS03/  - Klasifikasi dengan Naive Bayes
├── JS05/  - Regresi Linear & Polynomial
├── JS06/  - Support Vector Machine (SVM)
├── JS07/  - Decision Tree & Random Forest
├── JS09/  - Convolutional Neural Network (CNN)
├── JS10/  - Transfer Learning
├── JS11/  - Deep Learning Advanced
├── JS13/  - Natural Language Processing (NLP)
├── JS14/  - Text Classification & Sentiment Analysis
├── JS15/  - Advanced NLP dengan TF-IDF
└── README.md
```

---

## 📋 Detail Jobsheet

###  JS01 - Perkenalan Machine Learning & Python Basics
**File:** `TG1_2341720069_Ahmad Fadlih Wahyu Sardana.ipynb`

**Topik:**
- Pengenalan dasar Machine Learning
- Setup environment Python
- Library fundamental (NumPy, Pandas)
- Operasi dasar data manipulation

**Key Learning:**
- Memahami konsep dasar ML
- Instalasi dan konfigurasi tools
- Penggunaan Jupyter Notebook

---

###  JS02 - Visualisasi Data & Matplotlib
**File:** `TG1_JS02.ipynb`

**Topik:**
- Data visualization dengan Matplotlib
- Pandas untuk data manipulation
- Berbagai jenis plot (line, scatter, bar, histogram)
- Customization grafik
- Subplots dan multiple figures

**Key Learning:**
- Membuat visualisasi data yang informatif
- Exploratory Data Analysis (EDA)
- Interpretasi data melalui grafik
- Best practices dalam data visualization

**Visualisasi yang Dipelajari:**
- Line plots
- Scatter plots
- Bar charts
- Histograms
- Multiple subplots

---

### JS03 - Klasifikasi dengan Naive Bayes
**File:** `TG1_JS03_LAT_JSO3.ipynb`

**Topik:**
- Algoritma Naive Bayes Classifier
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Preprocessing data untuk klasifikasi
- Evaluasi model (accuracy, precision, recall, F1-score)
- Confusion Matrix

**Dataset:**
- Iris dataset
- Custom classification problems

**Key Learning:**
- Implementasi Naive Bayes dari scratch dan dengan scikit-learn
- Pemahaman probabilitas dalam machine learning
- Model evaluation metrics
- Train-test split methodology

---

###  JS05 - Regresi Linear & Polynomial
**File:** `TG1_JS05.ipynb`

**Topik:**
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Feature engineering
- Model evaluation (R², MSE, RMSE)
- Overfitting dan underfitting
- Regularization techniques

**Key Learning:**
- Memahami hubungan linear antar variabel
- Prediksi nilai kontinu
- Memilih model yang tepat
- Cross-validation
- Model optimization

**Teknik yang Dipelajari:**
- Least squares method
- Gradient descent
- Feature scaling
- Polynomial features transformation

---

### JS06 - Support Vector Machine (SVM)
**File:** `TG1_JS06.ipynb`

**Topik:**
- Support Vector Classification (SVC)
- Support Vector Regression (SVR)
- Kernel tricks (linear, polynomial, RBF, sigmoid)
- Hyperparameter tuning
- Decision boundaries
- Multi-class classification

**Key Learning:**
- Konsep margin dan support vectors
- Kernel selection
- Parameter optimization (C, gamma)
- Handling non-linear data
- Grid search untuk hyperparameter tuning

**Aplikasi:**
- Binary classification
- Multi-class classification
- Image classification basics

---

### JS07 - Decision Tree & Random Forest
**File:** `TG1_JS07.ipynb`

**Topik:**
- Decision Tree Classifier
- Decision Tree Regressor
- Random Forest Classifier
- Random Forest Regressor
- Feature importance
- Ensemble methods
- Bagging and boosting concepts

**Key Learning:**
- Tree-based algorithms
- Handling categorical and numerical data
- Preventing overfitting dengan pruning
- Ensemble learning advantages
- Feature selection techniques

**Evaluasi:**
- Cross-validation
- Out-of-bag error
- Variable importance analysis

---

### JS09 - Convolutional Neural Network (CNN)
**File:** `TG1_2_3_JS09.ipynb`

**Topik:**
- Introduction to Deep Learning
- Neural Network basics
- Convolutional Neural Networks (CNN)
- Image preprocessing dan augmentation
- CNN architecture design
- Convolutional layers, pooling layers
- Activation functions (ReLU, Sigmoid, Softmax)
- Training deep networks

**Dataset:**
- MNIST
- CIFAR-10
- Custom image datasets

**Key Learning:**
- Computer vision fundamentals
- Image classification dengan deep learning
- CNN architecture patterns
- Optimization techniques (Adam, SGD)
- Batch normalization
- Dropout for regularization

**Tools:**
- TensorFlow/Keras
- Image data generators
- Model callbacks

---

###  JS10 - Transfer Learning
**File:** `TG1_2_3_JS10.ipynb`

**Topik:**
- Transfer Learning concepts
- Pre-trained models (VGG16, ResNet, MobileNet, InceptionV3)
- Fine-tuning strategies
- Feature extraction
- Domain adaptation
- Model optimization

**Key Learning:**
- Leveraging pre-trained models
- Reducing training time
- Handling small datasets
- Custom top layers
- Freezing and unfreezing layers

**Aplikasi:**
- Image classification dengan limited data
- Multi-class classification
- Binary classification tasks

---

### JS11 - Deep Learning Advanced
**File:** `TG1_2_3_JS11.ipynb`

**Topik:**
- Advanced CNN architectures
- Object detection basics
- Semantic segmentation introduction
- Multi-task learning
- Model interpretation
- Advanced data augmentation
- Learning rate scheduling
- Custom loss functions

**Key Learning:**
- Complex model architectures
- Advanced optimization techniques
- Model visualization
- Performance tuning
- Handling imbalanced datasets

**Tools & Techniques:**
- TensorBoard visualization
- Model checkpointing
- Early stopping
- Learning rate finder
- Gradient visualization

---

### JS13 - Natural Language Processing (NLP)
**File:** `TG1_2_3_JS13.ipynb`

**Topik:**
- Text preprocessing
- Tokenization
- Stemming dan Lemmatization
- Bag of Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Word embeddings basics
- Text classification
- Feature extraction dari text

**Key Learning:**
- Memproses data text
- Text vectorization techniques
- NLP pipeline
- Feature engineering untuk text data
- Language modeling basics

**Preprocessing Techniques:**
- Lowercasing
- Removing punctuation
- Stop words removal
- Text normalization

---

### JS14 - Text Classification & Sentiment Analysis
**File:** `TG1_2_LAB_JS14.ipynb`

**Topik:**
- Sentiment Analysis
- Text Classification dengan Machine Learning
- Feature extraction (TF-IDF, Count Vectorizer)
- Classification algorithms untuk text
- Model evaluation untuk NLP
- Cross-validation pada text data

**Algoritma:**
- Naive Bayes untuk text
- Logistic Regression
- Support Vector Machines
- Random Forest untuk text classification

**Key Learning:**
- Sentiment classification (positive/negative)
- Multi-class text classification
- Handling text data
- NLP model evaluation metrics

**Dataset:**
- Product reviews
- Movie reviews
- Custom text datasets

---

### JS15 - Advanced NLP dengan TF-IDF
**File:** `TG1_2_LAB_JS15.ipynb`

**Topik:**
- Advanced TF-IDF techniques
- N-grams (unigrams, bigrams, trigrams)
- Feature selection untuk text
- Advanced text preprocessing
- Model comparison
- Hyperparameter tuning untuk NLP models

**Key Learning:**
- Optimizing TF-IDF parameters
- Feature engineering advanced
- Model selection untuk text tasks
- Performance optimization
- Production-ready NLP models

**Advanced Concepts:**
- Custom tokenizers
- Advanced vectorization
- Pipeline optimization
- Model deployment considerations

---

##  Technology Stack

### Programming Language
- **Python 3.x**

### Core Libraries
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

### Machine Learning
- **Scikit-learn** - Classical ML algorithms
- **TensorFlow** - Deep learning framework
- **Keras** - High-level neural networks API

### NLP
- **NLTK** - Natural Language Toolkit
- **spaCy** - Industrial-strength NLP
- **TextBlob** - Text processing

### Others
- **Jupyter Notebook** - Interactive development environment
- **Google Colab** - Cloud-based notebook platform

---

## Algoritma yang Dipelajari

### Supervised Learning
1. **Classification:**
   - Naive Bayes (Gaussian, Multinomial)
   - Support Vector Machines (SVM)
   - Decision Trees
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Logistic Regression

2. **Regression:**
   - Linear Regression (Simple & Multiple)
   - Polynomial Regression
   - Support Vector Regression (SVR)
   - Decision Tree Regression
   - Random Forest Regression

### Deep Learning
1. **Neural Networks:**
   - Artificial Neural Networks (ANN)
   - Convolutional Neural Networks (CNN)
   - Deep Neural Networks

2. **Architectures:**
   - VGG16
   - ResNet
   - MobileNet
   - InceptionV3

### Natural Language Processing
1. **Text Processing:**
   - Bag of Words (BoW)
   - TF-IDF
   - Word Embeddings
   
2. **Applications:**
   - Sentiment Analysis
   - Text Classification
   - Topic Modeling

---

## Konsep Machine Learning

### 1. Data Processing
- Data cleaning
- Feature engineering
- Feature scaling & normalization
- Handling missing values
- Data augmentation
- Train-test split
- Cross-validation

### 2. Model Development
- Model selection
- Hyperparameter tuning
- Grid search & Random search
- Regularization (L1, L2, Dropout)
- Ensemble methods
- Transfer learning

### 3. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve & AUC
- Mean Squared Error (MSE)
- R-squared (R²)
- Cross-validation scores
- Learning curves

### 4. Optimization Techniques
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Adam Optimizer
- Learning rate scheduling
- Batch normalization
- Early stopping

---


##  Learning Outcomes

Setelah menyelesaikan jobsheet-jobsheet ini, saya telah mampu:

1. ✅ Memahami konsep fundamental Machine Learning
2. ✅ Mengimplementasikan berbagai algoritma ML dari scratch dan dengan library
3. ✅ Melakukan preprocessing dan feature engineering
4. ✅ Membuat dan mengevaluasi model ML
5. ✅ Mengoptimalkan performa model
6. ✅ Mengimplementasikan Deep Learning untuk Computer Vision
7. ✅ Menerapkan Transfer Learning
8. ✅ Melakukan Text Processing dan NLP
9. ✅ Membangun model Sentiment Analysis
10. ✅ Memvisualisasikan dan menginterpretasikan hasil

---

## Resources & References

### Official Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Matplotlib Documentation](https://matplotlib.org/)