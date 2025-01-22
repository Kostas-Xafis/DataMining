
from sklearn.ensemble import HistGradientBoostingClassifier
best_model = HistGradientBoostingClassifier(
    class_weight='balanced',
    max_iter=200,
    max_depth=20,
    early_stopping=False,
    learning_rate=0.2,
    l2_regularization=0.2
)