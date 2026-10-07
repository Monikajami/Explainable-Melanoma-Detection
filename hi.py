import matplotlib.pyplot as plt

# ---- Your Real Metrics ----
accuracy = 77.31
precision = 5.71
recall = 75.28
f1_score = 10.62

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1_score]

plt.figure(figsize=(8,5))
bars = plt.bar(metrics, values)

plt.title("Performance Metrics of Proposed Model")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.2f}%',
             ha='center', va='bottom')

plt.savefig("performance_bar_graph.png", dpi=300)
plt.show()

print("✅ Performance graph saved as performance_bar_graph.png")