from joblib import load

class ModelS:

    def __init__(self):
        self.model = load("./models/modeloSVC.joblib")
        self.exactitud =0.82
        

    def make_predictions(self, data):

        result = self.model.predict(data)
        
        return result[0]
    
    def make_predictions_probability(self,data):
        result = self.model.predict_proba(data)
        return result

    def getExactitud(self):
        return self.exactitud
