import java.util.Arrays;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;

import weka.core.converters.ConverterUtils.DataSource;

public class c {
	
	private static final String CSV_TEST_FILE_PATH__2017_dataset = "D:\\SPYDER\\CICIDS-2017-Test-Data.csv";
	private static final String CSV_TEST_FILE_PATH__2018_dataset = "D:\\SPYDER\\CICIDS-2018-Test-Data.csv";

	public static void main(String[] args) throws Exception {
		 DataSource source = new DataSource(CSV_TEST_FILE_PATH__2018_dataset);
	        Instances data = source.getDataSet();
	        if (data.classIndex() == -1)
	            data.setClassIndex(data.numAttributes() - 1);

	        LibSVM svm = new LibSVM();	        
	        svm.buildClassifier(data);
			Evaluation eval3 = new Evaluation(data);
			eval3.evaluateModel(svm, data);
			System.out.println(eval3.toSummaryString());
			System.out.println(Arrays.deepToString(eval3.confusionMatrix()));	
	}
}
