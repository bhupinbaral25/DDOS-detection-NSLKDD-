from dashboard.modeling import ClassificationModel

def show_result(data, data_label, clf):
    '''This function train the classifier and gererate all the reports of the Classifier 
    '''
    clf_obj = ClassificationModel(data, data_label, clf)
    clf_obj.train_model()
    clf_obj.generate_confusion_matrix()
    return clf_obj.get_report()
 