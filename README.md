# K-Nearest Neighbors (KNN) Classification on the Iris Dataset  

## 📌 Project Overview  
This project demonstrates the use of the **K-Nearest Neighbors (KNN) algorithm** for classifying the **Iris dataset**. The script loads the dataset, preprocesses the data, trains a KNN model, and evaluates its performance using accuracy and classification metrics.  

## 🚀 Features  
- Loads the **Iris dataset** using `scikit-learn`.  
- Preprocesses the data with **feature scaling**.  
- Splits the data into **training and test sets**.  
- Trains a **KNN classifier** with `k=3` neighbors.  
- Evaluates model performance with **accuracy and a classification report**.  

## 🛠 Installation & Setup  
### **Prerequisites**  
Ensure you have **Python 3.7+** installed along with the following dependencies:  
```sh  
pip install numpy pandas scikit-learn  
```

### **Running the Script**  
1. Clone or download this repository.  
2. Open a terminal in the project directory.  
3. Run the script using:  
   ```sh  
   python KNN_Example_CIS324.py  
   ```  

## 📊 Expected Output  
The script will output the model's **accuracy** and a **classification report**, showing precision, recall, and F1-score for each class in the Iris dataset. Example:  

```
Accuracy ---> 0.97

              precision    recall    f1-score    support
   setosa        1.00        1.00        1.00        10
versicolor       1.00        0.89        0.94         9
virginica        0.92        1.00        0.96        11

accuracy        0.97        30
macro avg       0.97        0.96        0.97        30
weighted avg    0.97        0.97        0.97        30
```

## 🔍 Notes  
- The dataset is **automatically loaded** from `scikit-learn`, so no need to download it separately.  
- The **KNN model is trained with k=3 neighbors**, but you can experiment with different values for better performance.  
- The **dataset is standardized** using `StandardScaler()` to improve KNN performance.  

## 🏆 Future Improvements  
- Allow the user to **input a custom k-value** for KNN.  
- Experiment with **different distance metrics (e.g., Manhattan, Minkowski)**.  
- Implement **cross-validation** to fine-tune the model.  

## 📜 License  
This project is for educational purposes. Feel free to modify and extend it as needed!  
