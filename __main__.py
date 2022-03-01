import pandas as pd
import numpy as np
import yaml
from dashboard import ClassificationModel

def show_result(data, data_label, clf):
    '''This function train the classifier and gererate all the reports of the Classifier 
    '''
    clf_obj = ClassificationModel(data, data_label, clf)
    clf_obj.train_model()
    
    return clf_obj.get_report()
 