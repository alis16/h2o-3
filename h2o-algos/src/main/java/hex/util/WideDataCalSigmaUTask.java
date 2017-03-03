package hex.util;

import water.MRTask;

/** 
	* This class contains methods that are used by PCA, SVD or GLRM class to deal with wide datasets. 
	*
	* Created by wendycwong on 3/2/17. 
	**/

public static class WideDatatCalSigmaUTask extends MRTask<WideDatatCalSigmaUTask> {
		double[] _svec; 
		public double _sval; 
		public long _nobs;
}