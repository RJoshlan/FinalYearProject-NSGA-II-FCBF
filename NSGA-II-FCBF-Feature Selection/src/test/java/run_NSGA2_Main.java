package nsga2_Run;

import java.util.ArrayList;
import java.util.List;

import com.JoshlanRaposo.nsgaii.FCBF.MaximiseRelevance;
import com.debacharya.nsgaii.Configuration;
import com.debacharya.nsgaii.NSGA2;
import com.debacharya.nsgaii.objectivefunction.AbstractObjectiveFunction;
public class run_NSGA2_Main {
	
	public static void main(String[] args) {
		List<AbstractObjectiveFunction> objectives = new ArrayList<>();

		objectives.add(new MaximiseRelevance());		
		Configuration configuration = new Configuration(objectives);
		configuration.setGenerations(20);
		configuration.setPopulationSize(20);
		configuration.setChromosomeLength(78); 
			
		NSGA2 nsga2 = new NSGA2(configuration);
		nsga2.run();
	}
}