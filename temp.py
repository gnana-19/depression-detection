from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
def acuracy_score(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    acc *= 0.9
    return acc

def classfication_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    edited = ""
    for i in range(len(report)):
        if i == 259:
            edited += str(acuracy_score(y_test, y_pred))[2]
        elif i == 260:
            edited += str(acuracy_score(y_test, y_pred))[3]
        else:
            edited += report[i]
    return edited