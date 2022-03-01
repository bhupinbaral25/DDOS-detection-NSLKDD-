from dashboard.modeling import ClassificationModel

def show_result(data, data_label, clf, model_path):
    '''This function train the classifier and gererate all the reports of the Classifier 
    '''
    clf_obj = ClassificationModel(data, data_label, clf)
    clf_obj.train_model()
    clf_obj.generate_confusion_matrix()
    clf_obj.save_the_model(f"{model_path}fine_model_{str(clf)[0:10]}.pickle")
    return clf_obj.get_report()
 