from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend Agg không cần GUI
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from django.conf import settings

def home(request):
    return render(request, 'home.html')

def LAB1(request):
    plot_created = False
    
    if request.method == 'POST':
        X = np.array([180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169, 168, 175, 169, 171, 155, 158, 175, 165]).reshape(-1, 1)
        y = np.array([86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75, 56, 89, 45, 60, 60, 72]).reshape((-1, 1))

        X = np.insert(X, 0, 1, axis=1)

        # Tính toán theta sử dụng Phương trình Bình phương Nhỏ nhất
        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        x1 = 150
        y1 = theta[0] + theta[1] * x1
        x2 = 190
        y2 = theta[0] + theta[1] * x2

        # Vẽ biểu đồ
        plt.plot([x1, x2], [y1, y2], 'r-')
        plt.plot(X[:,1], y[:,0], 'bo')
        plt.xlabel('Chiều cao')
        plt.ylabel('Cân nặng')
        plt.title('Chiều cao và cân nặng của sinh viên VLU')

        # Kiểm tra xem thư mục media có tồn tại không, nếu không thì tạo mới
        if not os.path.exists('media'):
            os.makedirs('media')

        # Lưu biểu đồ
        plt.savefig(os.path.join('media', 'result_plot.png'))
        plt.close()  # Đóng plot sau khi lưu để tránh ghi đè

        plot_created = True

    return render(request, 'LAB1.html', {'plot_created': plot_created})

def LAB2_1(request):
    results = None

    if request.method == 'POST':
        # Path to the CSV file
        csv_path = os.path.join(settings.BASE_DIR, 'data', 'Education.csv')

        # Read the CSV file
        data = pd.read_csv(csv_path)

        # Prepare a preview of the data to display on the HTML page
        data_preview = data.head()


        #thay đổi chữ hoa thành chữ thường
        data['Text'] = data['Text'].str.lower() 

        # Convert labels to binary (Positive: 1, Negative: 0)
        data['Label'] = data['Label'].apply(lambda x: 1 if x == 'Positive' else 0)

        # Split the data into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Label'], test_size=0.2, random_state=42)

        # Train Gaussian Naive Bayes model
        gnb = GaussianNB()
        vectorizer = CountVectorizer()
        # Fit the vectorizer on the training data and transform both training and test data
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()

        # Train the model on the training data
        gnb.fit(X_train_vec, y_train)

        # Make predictions on the testing data
        y_pred = gnb.predict(X_test_vec)

        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Prepare results for rendering
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'data_preview': data_preview
        }

    return render(request, 'LAB2_1.html', {'results': results})

def LAB2_2(request):
    results = None

    if request.method == 'POST':
        # Path to the CSV file
        csv_path = os.path.join(settings.BASE_DIR, 'data', 'drug200.csv')

        
        data = pd.read_csv(csv_path)

        data_preview = data.head()
        # Chuyển đổi các biến phân loại thành số
        data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})  # F: 0, M: 1
        data['BP'] = data['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})  # LOW: 0, NORMAL: 1, HIGH: 2
        data['Cholesterol'] = data['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})  # NORMAL: 0, HIGH: 1

        # Đặt X là các đặc trưng và y là nhãn
        X = data.drop(columns=['Drug'])  # Dữ liệu đầu vào (tất cả các cột trừ 'Drug')
        y = data['Drug']  # Nhãn (cột 'Drug')

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Gaussian Naive Bayes model
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        # Make predictions
        y_pred = gnb.predict(X_test)

        # Measure accuracy
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Prepare results for rendering
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'data_preview': data_preview
        }

    return render(request, 'LAB2_2.html', {'results': results})
def LAB3_1(request):return render(request, 'LAB3_1.html')
def LAB3_2(request):return render(request, 'LAB3_2.html')
def LAB3_3(request):return render(request, 'LAB3_3.html')
def LAB4_1(request):return render(request, 'LAB4_1.html')
def LAB4_2(request):return render(request, 'LAB4_2.html')
def LAB4_3(request):return render(request, 'LAB4_3.html')