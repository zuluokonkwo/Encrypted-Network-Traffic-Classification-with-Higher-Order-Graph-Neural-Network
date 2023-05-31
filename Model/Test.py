# PREDICTION
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, roc_auc_score
import seaborn as sn
import seaborn as sns
import pandas as pd
import numpy as np
y_pred = []
y_true = []

# iterate over test data
for data in test_loader:
        output = model(data) # Feed Network
    
        output = (torch.max(torch.exp(output), 1)[1]).numpy()
        y_pred.extend(output) # Save Prediction
        
        label = (data.y).numpy()
        y_true.extend(label)
        

# constant for classes
classes = ('A', 'B', 'C', 'D', 'E')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / cf_matrix.astype(float).sum(axis=1), index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (12,12), dpi=100)
plt.title('Confusion matrix for nonVPN classes')
sns.set(font_scale = 2.5)
sn.heatmap(df_cm, annot=True, annot_kws={'size':28}, fmt='.3f')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.ylabel('True')
plt.yticks(rotation=45)
plt.savefig('data.png')
print (classification_report(y_true, y_pred, target_names=classes, digits=4))
