# Call lib
import beta
# Import necessary libraries
import tkinter as tk
from tkinter import ttk
from sklearn.metrics import accuracy_score

# Create a Tkinter window
window = tk.Tk()
window.title('Apple Quality Dataset Viewer')
window.configure(bg='#8B8378')  # Set background color

# Get data from beta.py
data = beta.Voting.data

# Create a Frame for the left side
left_frame = tk.Frame(window, bg='#FAEBD7')  # Set background color for the left frame
left_frame.pack(side=tk.LEFT, padx=10, pady=20)

# Create labels and entry widgets for user input
input_labels = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']
entry_boxes = []

for label_text in input_labels:
    label = tk.Label(left_frame, text=label_text, font=("Arial", 12), bg='#FAEBD7')  # Set background color for labels
    label.grid(row=input_labels.index(label_text), column=0, pady=5, sticky='w')
    
    entry_var = tk.DoubleVar()  # Use DoubleVar for numeric input
    entry = tk.Entry(left_frame, textvariable=entry_var, font=("Arial", 12))
    entry.grid(row=input_labels.index(label_text), column=1, pady=5)
    
    entry_boxes.append(entry_var)

# Create a button to trigger predictions
predict_button = tk.Button(left_frame, text="Predict Quality", command=lambda: predict_quality(entry_boxes), bg='#4CAF50', fg='white')  # Set button color
predict_button.grid(row=len(input_labels), column=0, columnspan=2, pady=10)

# Create a label to display the predicted quality
result_label = tk.Label(left_frame, text="", font=("Arial", 14), bg='#FAEBD7')  # Set background color for the label
result_label.grid(row=len(input_labels) + 1, column=0, columnspan=2, pady=10)

# Create a label to display accuracy information
accuracy_labels = tk.Label(left_frame, text="", font=("Arial", 10), bg='#FAEBD7')  # Set background color for the label
accuracy_labels.grid(row=len(input_labels) + 2, column=0, columnspan=2, pady=10)

# Function to predict quality based on user input
def predict_quality(entry_boxes):
    # Get user input from entry boxes
    user_input = [entry.get() for entry in entry_boxes]
    
    # Make predictions using the Voting Classifier
    predictions = beta.Voting.voting_clf.predict([user_input])
    
    # Get individual predictions from each classifier
    individual_predictions = {
        'Random Forest': beta.Voting.random_forest_model.predict([user_input])[0],
        'Logistic': beta.Voting.logistic_model.predict([user_input])[0],
        'Naive Bayes': beta.Voting.naive_bayes_model.predict([user_input])[0],
        'SVM': beta.Voting.svm_model.predict([user_input])[0],
        'Decision Tree': beta.Voting.decision_tree_model.predict([user_input])[0],
        'MLP': beta.Voting.mlp_model.predict([user_input])[0],
        'AdaBoost': beta.Voting.adaboost_model.predict([user_input])[0]
    }

    # Count the number of 'Good' and 'Bad' predictions
    good_count = sum(1 for label in individual_predictions.values() if label == 1)
    bad_count = sum(1 for label in individual_predictions.values() if label == 0)

    # Update the display based on the majority vote
    if good_count >= 4:
        predicted_label = 'Good'
    elif bad_count >= 4:
        predicted_label = 'Bad'
    else:
        predicted_label = 'Undecided'  # You can customize this based on your preference
    
    # Display the predicted quality
    result_label["text"] = f"Predicted Quality: {predicted_label}"

    # Display accuracy information for each classifier
    accuracy_labels["text"] = ""
    for model, prediction in individual_predictions.items():
        accuracy_label = 'Good' if prediction == 1 else 'Bad'
        accuracy_labels["text"] += f"{model} Prediction: {accuracy_label}\n"


# Create a Frame for the right side
right_frame = tk.Frame(window, bg='#ffffff')  # Set background color for the right frame
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create a Treeview widget
tree = ttk.Treeview(right_frame, style="mystyle.Treeview")
tree["columns"] = tuple(data.columns)

# Add columns to Treeview
for col in data.columns:
    tree.column(col, anchor="center", width=100)
    tree.heading(col, text=col, anchor="center")

# Insert data into Treeview with alternating row colors
for index, row in data.iterrows():
    if index % 2 == 0:
        tree.insert("", index, values=tuple(row), tags=('even',))
    else:
        tree.insert("", index, values=tuple(row), tags=('odd',))

# Configure tag colors
tree.tag_configure('even', background='#007FFF')
tree.tag_configure('odd', background='#00FFFF')

# Insert data into Treeview
for index, row in data.iterrows():
    tree.insert("", index, values=tuple(row))

# Style for Treeview
style = ttk.Style()
style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Arial', 10), rowheight=25)  # Set style for Treeview

# Pack the Treeview to the right frame
tree.pack(expand=tk.YES, fill=tk.BOTH)

# Run the Tkinter event loop
window.mainloop()
