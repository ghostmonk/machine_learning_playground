from models.DecisionTreeTrainingModel import DecisionTreeTrainingModel

training_data = DecisionTreeTrainingModel(file_path='data/iowa-train.csv')
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
training_data.set_up_training(prediction_target='SalePrice', feature_columns=feature_columns)
best_leaf_num = training_data.find_best_leaf_num()
mae = training_data.get_mean_error(max_leaf_nodes=best_leaf_num)
print(f"A mean error of {mae} was found with {best_leaf_num} leaf nodes")
