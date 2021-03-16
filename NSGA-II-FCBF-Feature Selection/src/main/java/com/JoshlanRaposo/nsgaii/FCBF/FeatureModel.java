package com.JoshlanRaposo.nsgaii.FCBF;

import java.util.Arrays;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FeatureModel {

	private static final String CSV_TRAIN_FILE_PATH_2017_dataset = "D:\\SPYDER\\CICIDS-2017-Train-Data.csv";
	private static final String CSV_TEST_FILE_PATH__2017_dataset = "D:\\SPYDER\\CICIDS-2017-Test-Data.csv";
	
	private static final String CSV_TRAIN_FILE_PATH_2018_dataset = "D:\\SPYDER\\CICIDS-2018-Train-Data.csv";
	private static final String CSV_TEST_FILE_PATH__2018_dataset = "D:\\SPYDER\\CICIDS-2018-Test-Data.csv";

	private int[] startDatasetList;
	
	private int[] chromosomeBasedFeatureSubset;
	
	private int n_of_Subset_Features;

	private Instances getData;
	
	private int [] featureIndices;
	
	private  int[] orderedFeatureList;

	private  double[] SUScoreforEachFeature;

	private  boolean f_hasClass;

	private  int f_classIndex;
	
	private  int n_of_features;
	
	private int n_of_instances;

	private  double threshold = -1;

	private  int n_FeaturesToSelect = -1;

	private  int cal_n_ToSelect = -1;

	private double[][] rankedFCBF;

	private  double[][] f_Best;
		
	private int n_of_classes;
	
	public int[] getstartDatasetList() {
		return startDatasetList;
	}

	public void setstartDatasetList(int[] startFeatureList) {
		this.startDatasetList = startFeatureList;
	}
	
	public int[] getChromosomeBasedFeatureSubset() {
		return chromosomeBasedFeatureSubset;
	}

	public void setChromosomeBasedFeatureSubset(int[] chromosomeBasedFeatureSubset) {
		this.chromosomeBasedFeatureSubset = chromosomeBasedFeatureSubset;
	}

	public int getN_of_Subset_Features() {
		return n_of_Subset_Features;
	}

	public void setN_of_Subset_Features(int n_of_Subset_Features) {
		this.n_of_Subset_Features = n_of_Subset_Features;
	}

	public Instances getGetData() throws Exception {
		return getData = DataSource.read(CSV_TRAIN_FILE_PATH_2017_dataset);
	}

	public void setGetData(Instances getData) {
		this.getData = getData;
	}

	public int[] getFeatureIndices() {
		return featureIndices;
	}

	public void setFeatureIndices(int[] featureIndices) {
		this.featureIndices = featureIndices;
	}

	public int[] getOrderedFeatureList() {
		return orderedFeatureList;
	}

	public void setOrderedFeatureList(int[] orderedFeatureList) {
		this.orderedFeatureList = orderedFeatureList;
	}

	public double[] getSUScoreforEachFeature() {
		return SUScoreforEachFeature;
	}

	public void setSUScoreforEachFeature(double[] SUScoreforEachFeature) {
		this.SUScoreforEachFeature = SUScoreforEachFeature;
	}

	public boolean isF_hasClass() {
		return f_hasClass;
	}

	public void setF_hasClass(boolean f_hasClass) {
		this.f_hasClass = f_hasClass;
	}

	public int getF_classIndex() {
		return f_classIndex;
	}

	public void setF_classIndex(int f_classIndex) {
		this.f_classIndex = f_classIndex;
	}

	public int getN_of_features() {
		return n_of_features;
	}

	public void setN_of_features(int n_of_features) {
		this.n_of_features = n_of_features;
	}

	public int getN_of_instances() {
		return n_of_instances;
	}

	public void setN_of_instances(int n_of_instances) {
		this.n_of_instances = n_of_instances;
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public int getN_FeaturesToSelect() {
		return n_FeaturesToSelect;
	}

	public void setN_FeaturesToSelect(int n_FeaturesToSelect) {
		this.n_FeaturesToSelect = n_FeaturesToSelect;
	}

	public int getN_of_classes() {
		return n_of_classes;
	}

	public void setN_of_classes(int n_of_classes) {
		this.n_of_classes = n_of_classes;
	}

	public int getCal_n_ToSelect() {
		if (n_FeaturesToSelect >= 0) {
			cal_n_ToSelect = n_FeaturesToSelect;
		}
		if (f_Best.length > 0 && f_Best.length < cal_n_ToSelect) {
			cal_n_ToSelect = f_Best.length;
		}
		return cal_n_ToSelect;
	}

	public void setCal_n_ToSelect(int cal_n_ToSelect) {
		this.cal_n_ToSelect = cal_n_ToSelect;
	}

	public double[][] getrankedFCBF() {
		return rankedFCBF;
	}

	public void setrankedFCBF(double[][] rankedFCBF) {
		this.rankedFCBF = rankedFCBF;
	}

	public double[][] getF_Best() {
		return f_Best;
	}

	public void setF_Best(double[][] f_Best) {
		this.f_Best = f_Best;
	}
	
	private boolean inStarting(int feat) {
		if ((f_hasClass == true) && (feat == f_classIndex)) {
			return true;
		}

		if (chromosomeBasedFeatureSubset == null) {
			return false;
		}

		for (int i = 0; i < chromosomeBasedFeatureSubset.length; i++) {
			if (chromosomeBasedFeatureSubset[i] == feat) {
				return true;
			}
		}

		return false;
	}
	
	public boolean getInStarting(int feat) {
		return inStarting(feat);
	}

	@Override
	public String toString() {
		return "FeatureModel [startDatasetList=" + Arrays.toString(startDatasetList) + ", chromosomeBasedFeatureSubset="
				+ Arrays.toString(chromosomeBasedFeatureSubset) + ", n_of_Subset_Features=" + n_of_Subset_Features
				+ ", getData=" + getData + ", featureIndices=" + Arrays.toString(featureIndices)
				+ ", orderedFeatureList=" + Arrays.toString(orderedFeatureList) + ", SUScoreforEachFeature="
				+ Arrays.toString(SUScoreforEachFeature) + ", f_hasClass=" + f_hasClass + ", f_classIndex="
				+ f_classIndex + ", n_of_features=" + n_of_features + ", n_of_instances=" + n_of_instances
				+ ", threshold=" + threshold + ", n_FeaturesToSelect=" + n_FeaturesToSelect + ", cal_n_ToSelect="
				+ cal_n_ToSelect + ", rankedFCBF=" + Arrays.toString(rankedFCBF) + ", f_Best=" + Arrays.toString(f_Best)
				+ ", n_of_classes=" + n_of_classes + "]";
	}
}
