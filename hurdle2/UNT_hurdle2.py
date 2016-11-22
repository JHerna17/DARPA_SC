# UNT hurdle 2 Solution

class UNT_hurdle2:
	"""UNT_hurdle2
	clf_path Path to where the clf is stored
	le_path Path to where the label encoder is stored
	"""
	from sklearn.externals import joblib
	import pandas as pd
	import scipy
	from scipy import signal
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.preprocessing import LabelEncoder




	def __init__(self, clf_path, le_path):
		self.clf = joblib.load(clf_path)
		self.le_l = joblib.load(le_path)
		
	def make_prediction(sample_file):
		iq_samples = scipy.fromfile(sample_file, dtype=scipy.complex64)
		fi, Pxi = PSD(iq_samples,3e6)	# Compute PSD
		tmpdf = ComputeFreqFeatures(Pxi,fi) # Feature Engineering
		return self.le_l.inverse_transform(predict_h2(tmpdf)) # Return predictions


	def read_samples_scipy(filename, blocklen):
    
	    iq_samples = scipy.fromfile(filename, 
	                                dtype=scipy.complex64, 
	                                count=blocklen)

    	return iq_samples

	def PSD(x, fs):
	    
	    f, Pxx_den = signal.periodogram(x, fs)
	    return f,Pxx_den


	def breakbins(PSD,f):
	    total_points = PSD.shape[0]
	    bins = np.split(PSD, 30)
	    fbins = np.split(f, 30)
	    keys = [x.min() for x in fbins]
	    df= pd.DataFrame(columns=['min_freq','freq','psd'])
	    for i in range(30):
	        df = df.append(pd.DataFrame([(keys[i],fbins[i],bins[i])],columns=['min_freq','freq','psd']),ignore_index=True)

	    df = df.sort('min_freq')
	    df['bin'] = range(30)
	    
	    return df

	def ComputeFreqFeatures(Px,f):
	    
	    df = breakbins(Px,f)
	    df.psd = df.psd.apply(lambda x: np.log(x))
	    df['PSDmax']= df.psd.apply(max)
	    df['PSDmin']= df.psd.apply(min)
	    df['PSDmean']= df.psd.apply(np.mean)
	    df['PSDstd']= df.psd.apply(np.std)
	    
	    return df
	def predict_h2(df):
    	return self.clf.predict(df[['PSDmax','PSDmin','PSDmean','PSDstd']])
