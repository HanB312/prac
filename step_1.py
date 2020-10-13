import pandas as pd
import scipy.stats as ss

def feature_pre(path_to_dir_csv):
	data = pd.read_csv(path_to_dir_csv)
	
	for i in ['age','weight','height']:
		# standardization
		std_feature = ss.zscore(data[i])
		data.insert(data.columns.get_loc(i),'std_'+i,std_feature)

		# normalization
		nor_feature = (data[i] - data[i].min(axis=0))/(data[i].max(axis=0) - data[i].min(axis=0))
		data.insert(data.columns.get_loc(i),'nor_'+i,nor_feature)

	return data

