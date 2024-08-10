import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
folder_path = base_dir.parent / 'data' 

HappinessScore_2022 = pd.read_csv(f'{folder_path}\World Happiness Report 2022.csv', encoding='utf-8')
HappinessScore_2023 = pd.read_csv(f'{folder_path}\World Happiness Report 2023.csv', encoding='utf-8')

# columns_2022 = list(HappinessScore_2022.columns)
# columns_2023 = list(HappinessScore_2023.columns)

# print(columns_2022)
# print(columns_2023)

HappinessScore_2023 = HappinessScore_2023.rename(columns = {
    'Country name': 'Country',
    'Ladder score': 'Happiness score',
    'upperwhisker': 'Whisker-high',
    'lowerwhisker': 'Whisker-low',
    'Dystopia + residual': 'Dystopia (1.83) + residual',
    'Explained by: Log GDP per capita': 'Explained by: GDP per capita',
    'Explained by: Perceptions of corruption': 'Explained by: Perceptions of corruption'
    }
    )

HappinessScore_2022 = HappinessScore_2022.iloc[:, 1:]
HappinessScore_2023 = HappinessScore_2023.loc[:, list(HappinessScore_2022.columns)]

HappinessScore_2022.to_csv(f'{folder_path}\data_2022.csv', index=False)
HappinessScore_2023.to_csv(f'{folder_path}\data_2023.csv', index=False)