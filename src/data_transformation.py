import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_transformation_module.data_transformation import DataTransformer


if __name__ == '__main__':
    obj3 = DataTransformer()
    WHR3 = obj3.columns_renamer('World Happiness Report 2023')
    
    obj2 = DataTransformer()
    WHR2 = obj2.data_load('World Happiness Report 2022')
    WHR2 = WHR2.iloc[:, 1:]
    
    WHR3 = WHR2.loc[:, list(WHR2.columns)]

    WHR2.to_csv(f'{obj2.folder_path}\data_2022.csv', index=False)
    WHR3.to_csv(f'{obj3.folder_path}\data_2023.csv', index=False) 
    