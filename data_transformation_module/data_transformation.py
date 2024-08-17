import pandas as pd
from pathlib import Path

class DataTransformer:
    base_dir = Path(__file__).resolve().parent
    folder_path = base_dir.parent / 'data'
    
    def data_load(self, file_name) -> pd.DataFrame:            
        data = pd.read_csv(f'{self.folder_path}\{file_name}.csv', encoding='utf-8')
        return data
    
    def columns_renamer(self, file_name) -> pd.DataFrame:
        names = {
            'Country name': 'Country',
            'Ladder score': 'Happiness score',
            'upperwhisker': 'Whisker-high',
            'lowerwhisker': 'Whisker-low',
            'Dystopia + residual': 'Dystopia (1.83) + residual',
            'Explained by: Log GDP per capita': 'Explained by: GDP per capita',
            'Explained by: Perceptions of corruption': 'Explained by: Perceptions of corruption'
            }
        
        data = self.data_load(file_name)
        data = data.rename(columns = names)
        
        return data