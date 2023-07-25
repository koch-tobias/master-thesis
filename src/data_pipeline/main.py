import pandas as pd

from preprocessing import preprocess_dataset
from data_analysis import plot_vehicle

# %%
def main():
    plot_bounding_boxes_one_vehicle = False
    plot_bounding_boxes_all_vehicle_by_name = False

    dataset_path_for_plot = "Path of dataset which should be plotted"

    if plot_bounding_boxes_one_vehicle:
        df = pd.read_excel(dataset_path_for_plot, index_col=0) 
        df = df[(df['X-Max'] != 0) & (df['X-Min'] != 0)]
        df = df[df["Relevant fuer Messung"] == "Ja"]
        unique_names = df["Einheitsname"].unique().tolist()
        unique_names.sort()
        for name in unique_names:
            print(name)
            df_new = df[(df["Einheitsname"] == name)]
            plot_vehicle(df_new, add_valid_space=True, preprocessed_data=False, mirrored=False)

    if plot_bounding_boxes_all_vehicle_by_name:
        df = pd.read_excel(dataset_path_for_plot,index_col=0) 
        df_preprocessed, df_for_plot = preprocess_dataset(df)
        plot_vehicle(df_for_plot, add_valid_space=True, preprocessed_data=False, mirrored=False)
    

# %%
if __name__ == "__main__":
    
    main()