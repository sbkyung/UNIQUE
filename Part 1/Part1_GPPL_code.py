import numpy as np
import pandas as pd
from pycaret.regression import *

#--------------------------------------------------------------------------

path_root = 'UNIQUE/Part 1/';

#--------------------------------------------------------------------------

dataset = pd.read_csv(path_root + 'Input/Part1_GPPL_example_input.csv')

#--------------------------------------------------------------------------

varind = ['TA_E','VPD_E','PAR_M',
          'NDVI_L','NIRV_L','LSWI_L','NIRVP_L']

#--------------------------------------------------------------------------

## Divide input
test_input = dataset[varind].copy()

## Load model
model = load_model(path_root + 'pretrained/Part1_GPPL_model')

## Prediction
test_result = predict_model(estimator = model,
                            data = test_input)
    
## Save result
test_result.to_csv(path_root + 'expected_output/Part1_GPPL_example_output.csv',index=False)
    
