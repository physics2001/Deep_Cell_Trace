# Deep Lineage

## Code Explanation

The folder structure and content are listed below. This repository provides the most preliminary code that are all python scripts. To run the scripts, the easiest way is to copy each script to the same folder to run. The first scripts to look at are in 📁Data_Generation folder which first will generate the required data for model training and testing. Then, we can look at the 📁Autoencoders folder to train autoencoders to produice encoded data. Next, 📁Classification_Model_Training_On_Hematopoesis, 📁Classification_Model_Training_On_Hematopoesis, 📁Regression_Model_Training_On_Hematopoesis, and 📁Regression_Model_Training_On_Reprogramming folders contain scripts to train different classification and regression models. After training the models, the saved models can be used with scripts in 📁Perturbation_On_Hematopoesis, 📁Perturbation_On_Reprogramming, and 📁Ablation folders to perform perturbation and ablation studies. 📁Plot folder contains scripts that can reproduce the plots used in the paper given the results are already generated. 

## Folder Structure and Content
```
└── 📁Deep_Cell_Trace
    └── 📁Ablation: Scripts related to ablation studies
        └── Ablate_Encoded_Hemato_Classification.py
        └── Ablate_Encoded_Reprogramming_Classification.py
        └── Gen_Encoded_Hematopoesis_Random_Time_Series_With_Monocyte_Neutrophil.py
        └── Gen_Encoded_Reprogramming_Random_Time_Series_With_Different_Preprocessing_Two_Classes.py
    └── 📁Autoencoders: Scripts on autoencoder training and encoding data
        └── Encode_Hematopoesis_Data.py
        └── Encode_Reprogramming_Data.py
        └── Train_Autoencoders_Hematopoesis.py
        └── Train_Autoencoders_Reprogramming.py
    └── 📁Classification_Model_Training_On_Hematopoesis: Scripts that train classification models on Hematorpoesis data
        └── Train_Classification_On_Encoded_Time_Series_Neu_Mon_Two_Classes.py
        └── Train_Classification_On_Encoded_Time_Series_Neu_Mon_Two_Classes_Without_Day4_And_Day6.py
        └── Train_Classification_On_Encoded_Time_Series_Neu_Mon_Two_Classes_Without_Day6.py
    └── 📁Classification_Model_Training_On_Reprogramming: Scripts that train classification models on Reprogramming data
        └── Train_Classification_On_Reprogramming_Encoded_Time_Series_Two_Classes.py
        └── Train_Classification_On_Reprogramming_Encoded_Time_Series_Two_Classes_Without_12_15_21_28.py
        └── Train_Classification_On_Reprogramming_Encoded_Time_Series_Two_Classes_Without_15_21_28.py
        └── Train_Classification_On_Reprogramming_Encoded_Time_Series_Two_Classes_Without_21_28.py
        └── Train_Classification_On_Reprogramming_Encoded_Time_Series_Two_Classes_Without_28.py
        └── Train_Classification_On_Reprogramming_Time_Series_Two_Classes.py
        └── Train_Classification_On_Reprogramming_Time_Series_Without_28.py
    └── 📁Data_Generation: Scripts that generate required time series for training
        └── Gen_Encoded_Time_Series_Hematopoesis.py
        └── Gen_Encoded_Time_Series_Reprogramming.py
        └── Gen_Hemotapoesis_Scaled_Logged_Data.py
        └── Gen_Hemotapoesis_Time_Series_Data.py
        └── Gen_Reprogramming_Encoded_Time_Series_Indices.py
        └── Gen_Reprogramming_Time_Series_Indices.py
        └── Gen_Reprogramming_Time_Series_With_Different_Preprocessing.py
    └── 📁Perturbation_On_Hematopoesis: Scripts related to perturbation studies on Hemotapoesis data
        └── Perturb_Hemato_Day4_Cells_Gened_Full_WIth_Reg.py
        └── Perturb_Hemato_Day4_Cells_Gened_Full_WIth_Reg_Not_Encoded.py
        └── Perturb_Hemato_Day4_Cells_Gened_Full_WIth_Reg_Not_Encoded_And_Compare_With_Actual_Cells.py
        └── Perturb_Hemato_Day4_Cells_Gened_Full_WIth_Reg_Not_Encoded_And_Compare_With_Actual_Cells_Neu_to_Mon.py
        └── Perturb_Hemo_Not_Encoded_Shap.py
    └── 📁Perturbation_On_Reprogramming: Scripts related to perturbation studies on Reprogramming data
        └── Perturb_Reprogramming_Day15_Cells_Gened_Full_WIth_Reg.py
        └── Perturb_Reprogramming_Day15_Cells_Gened_Full_WIth_Reg_Not_Encoded.py
        └── Perturb_Reprogramming_Day15_Cells_Gened_Full_WIth_Reg_Not_Encoded_And_Compare_With_Actual_Cells.py
        └── Perturb_Reprogramming_Day15_Cells_Gened_Full_WIth_Reg_Not_Encoded_Group_Contrast.py
        └── Perturb_Reprogramming_Day15_Cells_Gened_Full_WIth_Reg_Not_Encoded_Mult5.py
        └── Perturb_Reprogramming_Day15_Cells_Gened_Full_WIth_Reg_Not_Encoded_Mult5_Top_100.py
        └── Perturb_Reprogramming_Day15_Cells_Gened_Full_WIth_Reg_Not_Encoded_Top_100.py
        └── Perturb_Reprogramming_Not_Encoded_Shap.py
        └── Perturb_Reprogramming_Time_Series.py
    └── 📁Plot: Scripts that plot the data
        └── 📁.ipynb_checkpoints
            └── Figure_2_Plots-checkpoint.ipynb
        └── 📁ClassificationAccuracy: Plotting classification accuracy with given model less days of data
            └── Plot_Hemotapoesis_Classification_Accuracy_Effect_Of_Less_Days.py
            └── Plot_Reprogramming_Classification_Accuracy_Effect_Of_Less_Days.py
        └── 📁CorrelationPlots: Plotting correlation of average gene expressions of different datasets
            └── Plot_Gene_Expressed_Contrast_For_Cell_Types_Hematopoiesis.py
            └── Plot_Gene_Expressed_Contrast_For_Cell_Types_Reprogramming.py
        └── 📁DifferentPreprocessingResults: Plot different preprocessing's effect on autoencoders and regression models
            └── Plot_Preprocessing_Result.py
        └── Figure_2_Plots.ipynb
        └── 📁Heatmap: Plotting gene expression heatmap comparison between different class
            └── Plot_Gene_Expressed_Heatmap_Hematopoiesis.py
            └── Plot_Gene_Expressed_Heatmap_Reprogramming.py
        └── 📁RegressionCorrelation: Plotting average correlation over cells in testing
            └── Plot_Regression_Correlation_For_Different_Cell_Types_Hematopoiesis.py
            └── Plot_Regression_Correlation_For_Different_Cell_Types_Reprogramming.py
        └── 📁ROC: Plotting the receiver operating characteristic curves for classification models given different days of the data
            └── Plot_ROC_Curves_Hemato.py
            └── Plot_ROC_Curves_Reprogramming_Compared_WIth_CellRank.py
        └── 📁ViolinPlots: Plotting violin plots to compare predicted and actual distribution of the expression of each gene 
            └── Plot_Multiple_Violin_Plot_Hematopoiesis.py
            └── Plot_Multiple_Violin_Plot_Reprogramming.py
    └── 📁Regression_Model_Training_On_Hematopoesis: Scripts that train the regression models on hematopoesis data
        └── Train_Regression_On_Hemo_Time_Series_Neu_Mon_Day4.py
        └── Train_Regression_On_Hemo_Time_Series_Neu_Mon_Day6.py
    └── 📁Regression_Model_Training_On_Reprogramming: Scripts that train the reprogramming models on reprogramming data
        └── Train_Regression_On_Reprogramming_Time_Series_Two_Classes_Day12.py
        └── Train_Regression_On_Reprogramming_Time_Series_Two_Classes_Day15.py
        └── Train_Regression_On_Reprogramming_Time_Series_Two_Classes_Day15_With_All_Other_Days.py
        └── Train_Regression_On_Reprogramming_Time_Series_Two_Classes_Day21.py
        └── Train_Regression_On_Reprogramming_Time_Series_Two_Classes_Day21_With_All_Other_Days.py
        └── Train_Regression_On_Reprogramming_Time_Series_Two_Classes_Day28.py
```
