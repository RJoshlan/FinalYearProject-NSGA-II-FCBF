package com.JoshlanRaposo.nsgaii.FCBF;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;


public class SymmUncert {

	private static final double MAX_INT_Entropy_Funct = 10000d;
	private static final double[] INT_CACHE = new double[(int)MAX_INT_Entropy_Funct];
	  static {
		    for (int i = 1; i < MAX_INT_Entropy_Funct; i++) {
		      double d = (double)i;
		      INT_CACHE[i] = d * Math.log(d);
		    }
		  }

	private static FeatureModel featureModel = new FeatureModel();
	public static Instances getInstance;
	
	static {
		try {
			SymmUncert.control();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void control() throws Exception {
		getInstance = featureModel.getGetData();
		
		if (getInstance.classIndex() < 0) {
			getInstance.setClassIndex(getInstance.numAttributes()-1);
		}
		
		featureModel.setF_classIndex(getInstance.classIndex());
		featureModel.setN_of_features(getInstance.numAttributes());
		featureModel.setN_of_instances(getInstance.numInstances());
	    Discretize dtfs = new Discretize();
	    dtfs.setUseBetterEncoding(true);
	    dtfs.setBinRangePrecision(12);
	    dtfs.setInputFormat(getInstance);
	    getInstance = Filter.useFilter(getInstance, dtfs);
		featureModel.setN_of_classes(getInstance.attribute(featureModel.getF_classIndex()).numValues());

	}
	
	public static double lnFunc(double num) { // Natural Logarithm Function

		if (num <= 0) {
			return 0;
		} else {
			if (num < MAX_INT_Entropy_Funct) {
				int n = (int) num;
				if ((double) n == num) {
					return INT_CACHE[n];
				}
			}
			return num * Math.log(num);
		}
	}
	
	public static boolean eq(double a, double b) {

		return (a - b < 1e-6) && (b - a < 1e-6);
	}
	
	public static int partition(double[] array, int[] index, int fFirst, int lLast) {

		double pivot = array[index[(fFirst + lLast) / 2]];
		int help;

		while (fFirst < lLast) {
			while ((array[index[fFirst]] < pivot) && (fFirst < lLast)) {
				fFirst++;
			}
			while ((array[index[lLast]] > pivot) && (fFirst < lLast)) {
				lLast--;
			}
			if (fFirst < lLast) {
				help = index[fFirst];
				index[fFirst] = index[lLast];
				index[lLast] = help;
				fFirst++;
				lLast--;
			}
		}
		if ((fFirst == lLast) && (array[index[lLast]] > pivot)) {
			lLast--;
		}

		return lLast;
	}

	private static void quickSort(double[] array, int[] index, int left, int right) {

		if (left < right) {
			int middle = partition(array, index, left, right);
			quickSort(array, index, left, middle);
			quickSort(array, index, middle + 1, right);
		}
	}

	public static int[] sort(double[] array) {

		int[] index = new int[array.length];
		array = (double[]) array.clone();
		for (int i = 0; i < index.length; i++) {
			index[i] = i;
			if (Double.isNaN(array[i])) {
				array[i] = Double.MAX_VALUE;
			}
		}
		quickSort(array, index, 0, array.length - 1);
		return index;
	}

	public static double symmetric_Uncertainty(double vector[][]) {

		double sumOfColumn, sumOfRow, total = 0, cEntropy = 0, rEntropy = 0, conditionalEntropy = 0,
				mutualInformation = 0;

		// COLUMN ENTROPY
		for (int i = 0; i < vector[0].length; i++) {
			sumOfColumn = 0;
			for (int j = 0; j < vector.length; j++) {
				sumOfColumn += vector[j][i];
			}
			cEntropy += SymmUncert.lnFunc(sumOfColumn);
			total += sumOfColumn;
		}
		cEntropy -= SymmUncert.lnFunc(total);
		/*
		 * Calculates the conditional entropy H(X|Y) from two vectors. As a feature
		 * selection criterion, the best feature will maximise the mutual information
		 * MI(X, Y), where X = feature vector and Y = class indicator
		 */
		
		//ROW ENTROPY AND CONDITIONAL ENTROPY
		for (int i = 0; i < vector.length; i++) {
			sumOfRow = 0;
			for (int j = 0; j < vector[0].length; j++) {
				sumOfRow += vector[i][j];
				conditionalEntropy += SymmUncert.lnFunc(vector[i][j]);
			}
			rEntropy += SymmUncert.lnFunc(sumOfRow);
		}
		conditionalEntropy -= rEntropy;
		rEntropy -= SymmUncert.lnFunc(total);		
		mutualInformation = cEntropy - conditionalEntropy;
		
		double SU_X_Y = 2.0 * (mutualInformation / (cEntropy + rEntropy));
		if (SymmUncert.eq(cEntropy, 0) || SymmUncert.eq(rEntropy, 0)) {
			return 0;
		}
		return SU_X_Y;
	}
	
	/** evaluates an individual attribute by measuring the symmetrical uncertainty between it and the class*/
	public static double evaluateFeatures(int fi) throws Exception { // fi refers to individual feature
		Instance in;
		Instances getData = getInstance;
		int i, j, ii, jj;
		int ni, nj;
		double sum = 0.0;
		double sum_of_x[], sum_of_y[]; // x is i and j is y
		double SU_Value;
		boolean flag = false;
		boolean missingValues = true;
		double temp = 0.0;
		ni = getData.attribute(fi).numValues() + 1;
		nj = featureModel.getN_of_classes() + 1;// ******
		sum_of_x = new double[ni];
		sum_of_y = new double[nj];
		double[][] counts = new double[ni][nj];

		for (i = 0; i < ni; i++) {
			sum_of_x[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sum_of_y[j] = 0.0;
				counts[i][j] = 0.0;
			}
		}

		// Fill the contingency table
		for (i = 0; i < featureModel.getN_of_instances(); i++) {
			in = getData.instance(i);
			if (in.isMissing((int) fi)) {
				ii = ni - 1;
			} else {
				ii = (int) in.value((int) fi);
			}

			if (in.isMissing(featureModel.getF_classIndex())) {
				jj = nj - 1;
			} else {
				jj = (int) in.value(featureModel.getF_classIndex());
			}
			counts[ii][jj]++;
		}

		// get the row totals
		for (i = 0; i < ni; i++) {
			sum_of_x[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sum_of_x[i] += counts[i][j];
				sum += counts[i][j];
			}
		}

		// get the column totals
		for (j = 0; j < nj; j++) {
			sum_of_y[j] = 0.0;

			for (i = 0; i < ni; i++) {
				sum_of_y[j] += counts[i][j];
			}
		}

		// distribute missing counts
		if (missingValues && (sum_of_x[ni - 1] < featureModel.getN_of_instances()) && (sum_of_y[nj - 1] < featureModel.getN_of_instances())) {
			double[] i_copy = new double[sum_of_x.length];
			double[] j_copy = new double[sum_of_y.length];
			double[][] counts_copy = new double[sum_of_x.length][sum_of_y.length];

			for (i = 0; i < ni; i++) {
				System.arraycopy(counts[i], 0, counts_copy[i], 0, sum_of_y.length);
			}

			System.arraycopy(sum_of_x, 0, i_copy, 0, sum_of_x.length);
			System.arraycopy(sum_of_y, 0, j_copy, 0, sum_of_y.length);
			double total_missing = (sum_of_x[ni - 1] + sum_of_y[nj - 1] - counts[ni - 1][nj - 1]);

			// do the missing i's
			if (sum_of_x[ni - 1] > 0.0) {
				for (j = 0; j < nj - 1; j++) {
					if (counts[ni - 1][j] > 0.0) {
						for (i = 0; i < ni - 1; i++) {
							temp = ((i_copy[i] / (sum - i_copy[ni - 1])) * counts[ni - 1][j]);
							counts[i][j] += temp;
							sum_of_x[i] += temp;
						}

						counts[ni - 1][j] = 0.0;
					}
				}
			}

			sum_of_x[ni - 1] = 0.0;

			// do the missing j's
			if (sum_of_y[nj - 1] > 0.0) {
				for (i = 0; i < ni - 1; i++) {
					if (counts[i][nj - 1] > 0.0) {
						for (j = 0; j < nj - 1; j++) {
							temp = ((j_copy[j] / (sum - j_copy[nj - 1])) * counts[i][nj - 1]);
							counts[i][j] += temp;
							sum_of_y[j] += temp;
						}

						counts[i][nj - 1] = 0.0;
					}
				}
			}

			sum_of_y[nj - 1] = 0.0;

			// do the both missing
			if (counts[ni - 1][nj - 1] > 0.0 && total_missing != sum) {
				for (i = 0; i < ni - 1; i++) {
					for (j = 0; j < nj - 1; j++) {
						temp = (counts_copy[i][j] / (sum - total_missing)) * counts_copy[ni - 1][nj - 1];

						counts[i][j] += temp;
						sum_of_x[i] += temp;
						sum_of_y[j] += temp;
					}
				}

				counts[ni - 1][nj - 1] = 0.0;
			}
		}

		return SU_Value = symmetric_Uncertainty(counts);
	}
	
	/** Evaluates the feature sets by measuring the symmetric uncertainty between between sets of features */
	public static double evaluateFeaturesSet(double[] X, double[] Y) throws Exception { 
		// X = feature vector and Y = class indicator
		int i = 0, j = 0, ii = 0, jj = 0;
		Instance in;
		Instances getData = getInstance;
		int ni, nj, nni, nnj, N = featureModel.getN_of_instances();
		double sum = 0.0;
		double sum_of_x[], sum_of_y[];
		double[][] counts;
		double SU_Value;
		boolean flag = false;
		boolean missingValues = false;
		boolean missing_fi = false;									//fi refers to individual feature
		boolean missing_tv = false; 								//tv refers to target variable

		double temp = 0.0;
		int p;

		ni = (int) getData.attribute((int) X[0]).numValues();  				
		for (i = 1; i < X.length; i++) {
		      ni = ni*getData.attribute((int) X[i]).numValues();
		}
		ni+=1;
		
		nj = (int) getData.attribute((int) Y[0]).numValues(); 				
		for (i = 1; i < Y.length; i++) {
		      nj = nj*getData.attribute((int) Y[i]).numValues();
		}
		nj+=1;
		
		counts = new double[ni][nj];
		sum_of_x = new double[ni];
		sum_of_y = new double[nj];

		for (i = 0; i < ni; i++) {
			sum_of_x[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sum_of_y[j] = 0.0;
				counts[i][j] = 0.0;
			}
		}

		for (i = 0; i < N; i++) {
			in = getData.instance(i);
			missing_fi = false;
			missing_tv = false;
			nni = 1;
			ii = 0;

			for (p = Y.length - 1; p >= 0; p--) {
				if (in.isMissing((int) X[p])) {
					missing_fi = true;
				}
				ii = ((int) in.value((int) X[p]) * nni) + ii;
				if (p < X.length - 1) {
					nni = (int) (nni * getData.attribute((int) X[p]).numValues());						
				} else {
					nni = (int) (getData.attribute((int) X[p]).numValues());								
				}
			}
			if (missing_fi) {
				ii = ni - 1;
			}

			nnj = 1;
			jj = 0;
			for (p = Y.length - 1; p >= 0; p--) {
				if (in.isMissing((int) Y[p])) {
					missing_tv = true;
				}
				jj = ((int) in.value((int) Y[p]) * nnj) + jj;
				if (p < X.length - 1) {
					nnj = (int) (nnj * getData.attribute((int) Y[p]).numValues());						
				} else {
					nnj = (int) (getData.attribute((int) Y[p]).numValues());								
				}
			}
			if (missing_tv) {
				jj = nj - 1;
			}

			counts[ii][jj]++;
		}

		for (i = 0; i < ni; i++) {
			sum_of_x[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sum_of_x[i] += counts[i][j];
				sum += counts[i][j];
			}
		}

		for (j = 0; j < nj; j++) {
			sum_of_y[j] = 0.0;

			for (i = 0; i < ni; i++) {
				sum_of_y[j] += counts[i][j];
			}
		}

		// distribute missing counts
		if (missingValues && (sum_of_x[ni - 1] < N) && (sum_of_y[nj - 1] < N)) {
			double[] i_copy = new double[sum_of_x.length];
			double[] j_copy = new double[sum_of_y.length];
			double[][] counts_copy = new double[sum_of_x.length][sum_of_y.length];

			for (i = 0; i < ni; i++) {
				System.arraycopy(counts[i], 0, counts_copy[i], 0, sum_of_y.length);
			}

			System.arraycopy(sum_of_x, 0, i_copy, 0, sum_of_x.length);
			System.arraycopy(sum_of_y, 0, j_copy, 0, sum_of_y.length);
			double total_missing = (sum_of_x[ni - 1] + sum_of_y[nj - 1] - counts[ni - 1][nj - 1]);

			if (sum_of_x[ni - 1] > 0.0) {
				for (j = 0; j < nj - 1; j++) {
					if (counts[ni - 1][j] > 0.0) {
						for (i = 0; i < ni - 1; i++) {
							temp = ((i_copy[i] / (sum - i_copy[ni - 1])) * counts[ni - 1][j]);
							counts[i][j] += temp;
							sum_of_x[i] += temp;
						}

						counts[ni - 1][j] = 0.0;
					}
				}
			}

			sum_of_x[ni - 1] = 0.0;

			// do the missing j's
			if (sum_of_y[nj - 1] > 0.0) {
				for (i = 0; i < ni - 1; i++) {
					if (counts[i][nj - 1] > 0.0) {
						for (j = 0; j < nj - 1; j++) {
							temp = ((j_copy[j] / (sum - j_copy[nj - 1])) * counts[i][nj - 1]);
							counts[i][j] += temp;
							sum_of_y[j] += temp;
						}

						counts[i][nj - 1] = 0.0;
					}
				}
			}

			sum_of_y[nj - 1] = 0.0;

			// do the both missing
			if (counts[ni - 1][nj - 1] > 0.0 && total_missing != sum) {
				for (i = 0; i < ni - 1; i++) {
					for (j = 0; j < nj - 1; j++) {
						temp = (counts_copy[i][j] / (sum - total_missing)) * counts_copy[ni - 1][nj - 1];

						counts[i][j] += temp;
						sum_of_x[i] += temp;
						sum_of_y[j] += temp;
					}
				}

				counts[ni - 1][nj - 1] = 0.0;
			}
		}

		return SU_Value = symmetric_Uncertainty(counts);
	}
}
