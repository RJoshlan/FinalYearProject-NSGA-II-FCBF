package com.JoshlanRaposo.nsgaii.FCBF;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.debacharya.nsgaii.datastructure.Chromosome;
import com.debacharya.nsgaii.objectivefunction.AbstractObjectiveFunction;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeTransformer;
import weka.core.Instances;
import weka.core.Range;

public class MaximiseRelevance extends AbstractObjectiveFunction {

	private static Range startRange = new Range();
	private static FeatureModel featureModel = new FeatureModel();
	private static Instances getData;

	public MaximiseRelevance() {
		this.objectiveFunctionTitle = "Maximise-Relevance";
		try {
			getData = featureModel.getGetData();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public int[] getValue(Chromosome chromosome) {
		int[] featureSubset = null;
		int i;

		try {

			featureModel.setChromosomeBasedFeatureSubset(geneticCodeToDataset(chromosome));
			featureModel.setN_of_Subset_Features(featureModel.getChromosomeBasedFeatureSubset().length);

			int[] returnedFeatureSubset = calculatetheSUValueforEachFeature(
					featureModel.getChromosomeBasedFeatureSubset(), featureModel.getN_of_Subset_Features());

			featureSubset = new int[returnedFeatureSubset.length];

			for (i = 0; i < returnedFeatureSubset.length; i++) {
				featureSubset[i] = returnedFeatureSubset[i];
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return featureSubset;
	}
	
	public static int[] geneticCodeToDataset(Chromosome chromosome) throws Exception {

		List<String> geneList = chromosome.retrieveChromosomeGeneticCodes();
		getData = featureModel.getGetData();

		featureModel.setN_of_features(getData.numAttributes() - 1);
		featureModel.setFeatureIndices(new int[featureModel.getN_of_features()]);

		for (int i = 0, j = 0; i < featureModel.getN_of_features(); i++) {
			featureModel.getFeatureIndices()[j++] = i;
		}

		int[] attributeindex = featureModel.getFeatureIndices();
		ArrayList<Integer> x = new ArrayList<Integer>(); // Create selected feature list

		for (int i = 0; i < attributeindex.length; i++) {
			if (geneList.get(i) == "1") { // Referencing to dataset indices with gene value of 1
				x.add(attributeindex[i]);
			}
		}

		int[] getSelectedList = convertIntegers(x);

		return getSelectedList;
	}
	
	public static int[] calculatetheSUValueforEachFeature(int[] featureSubset, int n_of_Subset_Features)
			throws Exception {
		ASEvaluation asEvaluation = null;
		
		Instances getData = featureModel.getGetData();
		int i, j;
		
		if (getData.classIndex() < 0) {
			getData.setClassIndex(getData.numAttributes()-1);
		}
		
		featureModel.setF_classIndex(getData.classIndex());
		featureModel.setN_of_features(n_of_Subset_Features);
		featureModel.setN_of_instances(getData.numInstances());
		featureModel.setN_of_classes(getData.attribute(featureModel.getF_classIndex()).numValues());

		if (asEvaluation instanceof AttributeTransformer) {
			getData = ((AttributeTransformer) asEvaluation).transformedHeader();
			if (featureModel.getF_classIndex() >= 0 && getData.classIndex() >= 0) {
				featureModel.setF_classIndex(getData.classIndex());
				featureModel.setF_hasClass(true);
			}
		}

		startRange.setUpper(featureModel.getN_of_features());
		if (!(getStartSet().equals(""))) {
			featureModel.setstartDatasetList(featureModel.getChromosomeBasedFeatureSubset());
		}

		int counter = 0;

		if (featureModel.getstartDatasetList() != null) {
			counter = featureModel.getstartDatasetList().length;
		}

		if ((featureModel.getstartDatasetList() != null) && (featureModel.isF_hasClass() == true)) {
			boolean ok = false;
			for (i = 0; i < counter; i++) {
				if (featureModel.getstartDatasetList()[i] == featureModel.getF_classIndex()) {
					ok = true;
					break;
				}
			}

			if (ok == false) {
				counter++;
			}
		} else {
			if (featureModel.isF_hasClass() == true) {
				counter++;
			}
		}

		featureModel.setOrderedFeatureList(featureModel.getChromosomeBasedFeatureSubset());
		featureModel.setSUScoreforEachFeature(new double[featureModel.getN_of_features() - counter]);

		for (i = 0, j = 0; i < featureModel.getN_of_features();i++) {
			if (featureModel.getInStarting(i)) {
				featureModel.getOrderedFeatureList()[j++] = i;
			}
		}

		j = i = featureModel.getChromosomeBasedFeatureSubset().length;
		for (i = 0; i < featureModel.getOrderedFeatureList().length; i++) {
			featureModel.getSUScoreforEachFeature()[i] = 
					SymmUncert.evaluateFeatures(featureModel.getOrderedFeatureList()[i]);
		}

		double[][] tempRanked = sortEvalAttributes();
		int[] rankedAttributes = new int[featureModel.getF_Best().length];

		for (i = 0; i < featureModel.getF_Best().length; i++) {
			rankedAttributes[i] = (int) tempRanked[i][0];
		}

		return rankedAttributes;
	}

	/** Sorts the evaluated attribute list in descending order */
	public static double[][] sortEvalAttributes() throws Exception {
		int i, j;

		if (featureModel.getOrderedFeatureList() == null || featureModel.getSUScoreforEachFeature() == null) {
			throw new Exception("List empty. Cannot retrieve ranked attribute list");
		}

		int[] ranked = SymmUncert.sort(featureModel.getSUScoreforEachFeature());
		double[][] descendingOrder = new double[ranked.length][2];

		for (i = ranked.length - 1, j = 0; i >= 0; i--) {
			descendingOrder[j++][0] = ranked[i];
		}

		// Get the indexes of attribute
		for (i = 0; i < descendingOrder.length; i++) {
			int temp = ((int) descendingOrder[i][0]);
			descendingOrder[i][0] = featureModel.getOrderedFeatureList()[temp]; // for the index
			descendingOrder[i][1] = featureModel.getSUScoreforEachFeature()[temp]; // for the value of the index
		}

		if (featureModel.getN_FeaturesToSelect() > descendingOrder.length) {
			throw new Exception("More attributes requested than exist in the data");
		}
		
		removeRedundantFeatures(descendingOrder);

		if (featureModel.getN_FeaturesToSelect() <= 0) {
			if (featureModel.getThreshold() == -Double.MAX_VALUE) {
				featureModel.setCal_n_ToSelect(featureModel.getF_Best().length);
			} else {
				determineNumToSelectFromThreshold(featureModel.getF_Best());
			}
		}
		return featureModel.getF_Best();
	}

	public static void determineNumToSelectFromThreshold(double[][] ranking) {
		int count = 0;
		for (int i = 0; i < ranking.length; i++) {
			if (ranking[i][1] > featureModel.getThreshold()) {
				count++;
			}
		}
		featureModel.setCal_n_ToSelect(count);
	}

	/*
	 * 2nd stage starts here to remove redundant features and only keeps predominant
	 * ones among all the selected relevant features
	 */

	public static void removeRedundantFeatures(double[][] rankedFeatures) throws Exception {
		int i, j;

		featureModel.setrankedFCBF(new double[featureModel.getOrderedFeatureList().length][4]);
		double[] features = new double[1];
		double[] classFeatures = new double[1];

		int n_of_SelectedFeatures = 0;
		int startPoint = 0;
		double tempSUIJ = 0;

		for (i = 0; i < rankedFeatures.length; i++) {
			featureModel.getrankedFCBF()[i][0] = rankedFeatures[i][0];
			featureModel.getrankedFCBF()[i][1] = rankedFeatures[i][1];
			featureModel.getrankedFCBF()[i][2] = -1;
		}

		while (startPoint < rankedFeatures.length) {
			if (featureModel.getrankedFCBF()[startPoint][2] != -1) {
				startPoint++;
				continue;
			}

			featureModel.getrankedFCBF()[startPoint][2] = featureModel.getrankedFCBF()[startPoint][0];
			n_of_SelectedFeatures++;

			for (i = startPoint + 1; i < featureModel.getOrderedFeatureList().length; i++) {
				if (featureModel.getrankedFCBF()[i][2] != -1) {
					continue;
				}
				features[0] = (int) featureModel.getrankedFCBF()[startPoint][0];
				classFeatures[0] = (int) featureModel.getrankedFCBF()[i][0];
				tempSUIJ = SymmUncert.evaluateFeaturesSet(features, classFeatures);// measuring symUn
				if (featureModel.getrankedFCBF()[i][1] < tempSUIJ
						|| Math.abs(tempSUIJ - featureModel.getrankedFCBF()[i][1]) < 1E-8) {
					featureModel.getrankedFCBF()[i][2] = featureModel.getrankedFCBF()[startPoint][0];
					featureModel.getrankedFCBF()[i][3] = tempSUIJ;
				}
			}
			startPoint++;
		}

		featureModel.setF_Best(new double[n_of_SelectedFeatures][2]);

		for (i = 0, j = 0; i < featureModel.getOrderedFeatureList().length; i++) {
			if (featureModel.getrankedFCBF()[i][2] == featureModel.getrankedFCBF()[i][0]) {
				featureModel.getF_Best()[j][0] = featureModel.getrankedFCBF()[i][0];
				featureModel.getF_Best()[j][1] = featureModel.getrankedFCBF()[i][1];
				j++;
				
			}
		}
	}

	public void setStartSet(String startSet) throws Exception {
		startRange.setRanges("0,1,2");
	}

	public static String getStartSet() {
		return startRange.getRanges();
	}

	public static int[] convertIntegers(List<Integer> x) {
		int[] attriIntArr = new int[x.size()];

		for (int i = 0; i < attriIntArr.length; i++) {
			attriIntArr[i] = x.get(i).intValue();
		}

		return attriIntArr;

	}
}