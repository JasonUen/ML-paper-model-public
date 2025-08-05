"""
Data configuration for loading CSV files as DataFrames.

Each dictionary specifies:
- name: variable name for the loaded data
- relpath: user-specified sub-path to the CSV file
- columns: columns of interest (optional)
- rename: column renaming map (optional)
- drop_index: index to drop  (optional)
"""

data_configs = [
    {
        "name": "fw",  # Food waste data from the digital dataset
        "relpath": "EPA/ExcessFoodPublic_USTer_2018_R9/Output_dig/FW_digital.csv",
        "columns": ["GEOID", "MeanEF"],
    },
    {
        "name": "amp",  # USDA animal product dataset
        "relpath": "USDA/AnimalProducts/AnimalProduct.csv",
        "columns": ["CountyFIPS", "Value"],
        "rename": {"CountyFIPS": "GEOID", "Value": "AmlProduct"},
    },
    {
        "name": "crop",  # USDA crop dataset
        "relpath": "USDA/Crop/CropData.csv",
        "columns": ["GEOID", "Crop"],
    },
    {
        "name": "inc",  # Per capita income by county (US Census)
        "relpath": "US Census/IncomePerCapita/IncData.csv",
        "columns": ["GEOID", "2018"],
        "rename": {"2018": "IncPerCap"},
    },
    {
        "name": "pop",  # Population by county (US Census)
        "relpath": "US Census/Population/PopData2018.csv",
        "columns": ["CountyFIPS", "2018"],
        "rename": {"CountyFIPS": "GEOID", "2018": "Population"},
        "drop_index": 0,
    },
    {
        "name": "n_groc",  # Number of grocery stores
        "relpath": "USDA ERS/Output/n_groc.csv",
    },
    {
        "name": "n_superc",  # Number of supercenters & club stores
        "relpath": "USDA ERS/Output/n_superc.csv",
    },
    {
        "name": "n_conv",  # Number of convenience stores
        "relpath": "USDA ERS/Output/n_conv.csv",
    },
    {
        "name": "n_specs",  # Number of specialized food stores
        "relpath": "USDA ERS/Output/n_specs.csv",
    },
    {
        "name": "n_ffr",  # Number of fast-food restaurants
        "relpath": "USDA ERS/Output/n_ffr.csv",
    },
    {
        "name": "n_fsr",  # Number of full-service restaurants
        "relpath": "USDA ERS/Output/n_fsr.csv",
    },
    {
        "name": "MEDHHINC",  # Median household income
        "relpath": "USDA ERS/Output/MEDHHINC.csv",
        "columns": ["GEOID", "MEDHHINC"],
    },
    {
        "name": "PC_FFRSALES",  # Expenditures per capita -- fast food
        "relpath": "USDA ERS/Output/PC_FFRSALES.csv",
    },
    {
        "name": "PC_FSRSALES",  # Expenditures per capita -- full-service restaurants
        "relpath": "USDA ERS/Output/PC_FSRSALES.csv",
    },
    {
        "name": "SNAPS", # SNAP-authorized stores
        "relpath": "USDA ERS/Output/SNAPS.csv"},  
    {
        "name": "WICS", # WIC-authorized stores
        "relpath": "USDA ERS/Output/WICS.csv"},  
    {
        "name": "REDEMP_SNAPS",  # SNAP redemptions/SNAP-authorized stores
        "relpath": "USDA ERS/Output/REDEMP_SNAPS.csv",
    },
    {
        "name": "REDEMP_WICS",  # WIC redemptions/WIC-authorized stores
        "relpath": "USDA ERS/Output/REDEMP_WICS.csv",
    },
    {
        "name": "LACCESS_POP",  # Population with low access to store
        "relpath": "USDA ERS/Output/LACCESS_POP.csv",
        "columns": ["GEOID", "LACCESS_POP"],
    },
    {
        "name": "LACCESS_LOWI",  # Low income & low access to store
        "relpath": "USDA ERS/Output/LACCESS_LOWI.csv",
        "columns": ["GEOID", "LACCESS_LOWI"],
    },
    {
        "name": "LACCESS_HHNV",  # Households with no vehicle and low access to store
        "relpath": "USDA ERS/Output/LACCESS_HHNV.csv",
        "columns": ["GEOID", "LACCESS_HHNV"],
    },
    {
        "name": "LACCESS_SNAP",  # SNAP households with low access to store
        "relpath": "USDA ERS/Output/LACCESS_SNAP.csv",
        "columns": ["GEOID", "LACCESS_SNAP"],
    },
]
