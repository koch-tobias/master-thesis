paths = {
        "model_folder": "data/labeled"
        }

settings = {
            "prediction_threshold": 0.75
            }

general_params = {
                "relevant_features": ['Sachnummer','Benennung (dt)', 'X-Min','X-Max','Y-Min','Y-Max','Z-Min','Z-Max', 'Wert','Einheit','Gewichtsart','Kurzname','L-Kz.', 'L/R-Kz.', 'Modul (Nr)', 'ox','oy', 'oz', 'xx','xy','xz', 'yx','yy','yz','zx','zy','zz']  # List of features which are relevant for the models - all other features will be deleted
                }
