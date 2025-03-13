# **Robot Inverse Kinematics Prediction using Random Forest**

This project demonstrates the use of a **Random Forest Regressor** to predict the **end-effector position** \((x, y, z)\) based on **robot joint angles** \((q1, q2, q3, q4, q5, q6)\). The dataset consists of recorded joint angles and their corresponding end-effector positions.

## **Dataset**
- The dataset is a CSV file containing:
  - **Inputs (Features)**: Joint angles \(`q1, q2, q3, q4, q5, q6`\).
  - **Outputs (Targets)**: End-effector position \(`x, y, z`\).
- The dataset should be placed in the `data/` directory or updated in the script.

## **Installation**
Ensure you have Python installed along with the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## **Usage**
1. Place your dataset (`robot_inverse_kinematics_dataset.csv`) in the appropriate directory.
2. Update the file path in the script if needed.
3. Run the script:

```bash
python inverse_kinematics_prediction.py
```

## **Workflow**
1. **Load Data**: Reads the CSV file into a Pandas DataFrame.
2. **Preprocess Data**:
   - Drops missing values.
   - Defines features (`X`) and targets (`y`).
   - Splits data into **training (80%)** and **testing (20%)** sets.
3. **Train Model**:
   - Uses a **Random Forest Regressor** with `100` estimators.
   - Fits the model on the training set.
4. **Evaluate Model**:
   - Predicts the end-effector positions on the test set.
   - Computes **Mean Squared Error (MSE)**.
   - Plots **Predicted vs. Actual** values for visualization.

## **Results**
- The **Mean Squared Error (MSE)** is printed to assess model performance.
- A scatter plot visualizes the predicted vs. actual values of the **x-coordinate**.

## **Future Improvements**
- Implement feature scaling (e.g., **StandardScaler**).
- Test different regression models (e.g., **Neural Networks, SVR**).
- Use hyperparameter tuning (e.g., **GridSearchCV**).
- Extend the model to include inverse kinematics solvers.

## **License**
This project is open-source under the **MIT License**.
