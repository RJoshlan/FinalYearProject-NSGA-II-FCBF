/*
 * MIT License
 *
 * Copyright (c) 2019 Debabrata Acharya
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.debacharya.nsgaii;

import com.JoshlanRaposo.nsgaii.FCBF.FeatureModel;
import com.debacharya.nsgaii.datastructure.AbstractAllele;
import com.debacharya.nsgaii.datastructure.Chromosome;
import com.debacharya.nsgaii.datastructure.Population;
import com.debacharya.nsgaii.objectivefunction.AbstractObjectiveFunction;

import weka.core.Instances;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public class Reporter {

	private static int fileHash = ThreadLocalRandom.current().nextInt(10000, 100000);
	private static StringBuilder writeContent = new StringBuilder();

	public static boolean silent = false;
	public static boolean autoTerminate = true;
	public static boolean plotGraph = true;
	public static boolean plotCompiledGraphForEveryGeneration = true;
	public static boolean plotGraphForEveryGeneration = false;
	public static boolean writeToDisk = true;
	public static boolean diskWriteSuccessful = true;
	public static String outputDirectory = "Results";
	public static String filename = "NSGA-II-FCBF report-" + Reporter.fileHash + ".txt";
	public static FeatureModel featureModel = new FeatureModel();
	public static Instances getData;
	static {
		try {
			getData = featureModel.getGetData();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void init(Configuration configuration) {

		if(silent && !writeToDisk) return;
		if(!Reporter.outputDirectory.isEmpty() && !Reporter
													.outputDirectory
													.substring(Reporter.outputDirectory.length() - 1)
													.equals(File.separator))
			Reporter.outputDirectory += File.separator;

		p("\n[ " + java.time.LocalDateTime.now() + " ]");
		p(
		"** Output is" 											+
			(Reporter.writeToDisk ? "" : " not") 						+
			" written to disk" 											+
			(Reporter.writeToDisk ? (". " + Reporter.outputDirectory + Reporter.filename) : "") 	+
			"."
		);

		p("---------------------------------------------------------------------------------");
		p("   NON-DOMINATED SORTING GENETIC ALGORITHM-II - FAST CORRELATION BASED FILTER    ");
		p("   				NSGA-II-FCBF													");
		p("---------------------------------------------------------------------------------");
		p(configuration.toString());
	}

	public static void reportGeneration(Population parent, Population child, int generation, List<AbstractObjectiveFunction> objectives) {

		if(silent && !writeToDisk) return;

		p("\n++++++++++++++++ GENERATION: " + generation + " ++++++++++++++++\n");
		p("[ START ]\n");
		p("Parent Population: " + parent.size());
		p("Child Population: " + child.size());
		p("\n======== PARENT ========\n");
		Reporter.reportPopulation(parent);
		p("\n======== CHILD ========\n");
		Reporter.reportPopulation(child);
		p("\n[ END ]");
	}

	public static void reportPopulation(Population population) {

		if(silent && !writeToDisk) return;

		for(Chromosome chromosome : population.getPopulace()) {
			Reporter.reportChromosome(chromosome);
		}
		
	}

	public static void reportChromosome(Chromosome chromosome) {

		if(silent && !writeToDisk) return;

		Reporter.reportGeneticCode(chromosome.getGeneticCode());
		Reporter.reportChromosomeRefToDataset(chromosome);
		p(">> " + chromosome.toString());
	}

	public static void reportGeneticCode(List<AbstractAllele> geneticCode) {

		if(silent && !writeToDisk) return;

		StringBuilder code = new StringBuilder("## Chromosome [");

		for(AbstractAllele allele : geneticCode)
			code.append(allele.toString()).append(",");

		p(code.append("]").toString());
	}
	
	public static void reportChromosomeRefToDataset(Chromosome chromosome) {
		if (silent && !writeToDisk)
			return;

		StringBuilder code = new StringBuilder("## Dataset Index: ");
		try {
			List<String> newList = geneticCodeToDataset(chromosome);
			code.append(newList.toString()).append(",");
			p(code.append("").toString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static List<String> geneticCodeToDataset(Chromosome chromosome) throws Exception {

		List<String> geneList = chromosome.retrieveChromosomeGeneticCodes();

		featureModel.setN_of_features(getData.numAttributes() - 1);
		featureModel.setFeatureIndices(new int[featureModel.getN_of_features()]);

		for (int i = 0, j = 0; i < featureModel.getN_of_features(); i++) {
			featureModel.getFeatureIndices()[j++] = i;
		}

		int[] attributeindex = featureModel.getFeatureIndices();
		List<String> x = new ArrayList<String>(); 

		for (int i = 0; i < attributeindex.length; i++) {
			if (geneList.get(i) == "1") {
				x.add(String.valueOf(attributeindex[i]));
			}
		}

		return x;
	}

	public static void reportConcreteGeneticCode(List<? extends AbstractAllele> geneticCode) {

		if(silent && !writeToDisk) return;

		Reporter.reportGeneticCode(geneticCode.stream().map(e -> (AbstractAllele)e).collect(Collectors.toList()));
	}

	public static void plot2DParetoFront(Population population, List<AbstractObjectiveFunction> objectives) {
		Population paretoParent = getFinalParetoFront(population);
        p("---PARETO FRONT---");
    	reportPopulation(paretoParent);
	}

	public static void plotGraphs(Population finalChild, List<AbstractObjectiveFunction> objectives) {
		if(plotGraph) Reporter.plot2DParetoFront(finalChild, objectives);
	}

	public static void terminate(Population finalChild, List<AbstractObjectiveFunction> objectives) {

		Reporter.plotGraphs(finalChild, objectives);

		if(silent && !writeToDisk) return;

		p("------------------------------------------------");
		p("NSGA-II ENDED SUCCESSFULLY\n");
	}

	public static void commitToDisk() {
		if(writeToDisk) {
			Reporter.writeToFile();
			if(diskWriteSuccessful) p("** Output saved at " + outputDirectory + filename + "\n");
		}
	}

	public static void flush() {
		Reporter.writeContent = new StringBuilder();
		Reporter.fileHash = ThreadLocalRandom.current().nextInt(10000, 100000);
		Reporter.filename = "NSGA-II-report-" + Reporter.fileHash + ".txt";
	}

	public static void p(Object s) {
		if(writeToDisk) Reporter.writeContent.append(s).append(System.lineSeparator());
		if(!silent) System.out.println(s.toString());
	}

	private static void writeToFile() {

		try {

			Files.createDirectories(Paths.get(Reporter.outputDirectory));
			FileWriter writer = new FileWriter(Reporter.outputDirectory + Reporter.filename);

			writer.write(Reporter.writeContent.toString());
			writer.close();
		} catch (Exception e) {
			diskWriteSuccessful = false;
			System.out.println("\n!!! COULD NOT WRITE FILE TO DISK!\n\n");
		}
	}
	
	private static Population getFinalParetoFront(Population finalParent) {
		SortedMap<Integer, Chromosome> generationChromosomes = Service
				.returnSortedChromosomes(finalParent.getPopulace());
		SortedMap<Integer, Chromosome> paretoChromosomes = new TreeMap<>();

		Chromosome bestSolution = generationChromosomes.get(generationChromosomes.firstKey());
		Chromosome secndBestSolution = generationChromosomes.get(generationChromosomes.lastKey());
		paretoChromosomes.put(bestSolution.getObjectiveValues().get(0), bestSolution);
		paretoChromosomes.put(secndBestSolution.getObjectiveValues().get(0), secndBestSolution);

		List<Chromosome> generationChromosomeList = new ArrayList(generationChromosomes.values());

		for (int i = generationChromosomes.size() - 1; i >= 0; i--) {
			Chromosome chromosome = generationChromosomeList.get(i);
			if (dominatesInAll2Dimensions(chromosome, paretoChromosomes)) {
				List<Integer> objectiveValues = chromosome.getObjectiveValues();
				int xValueChromosome = objectiveValues.get(0);
				paretoChromosomes.put(xValueChromosome, chromosome);

			}
		}

		return new Population(new ArrayList<>(paretoChromosomes.values()));
	}

    private static boolean dominatesInAll2Dimensions(Chromosome chromosomeToCheck, SortedMap<Integer, Chromosome> paretoChromosomes) {
        boolean isDominatingInAllDimensions = false;
        List<Integer> objectiveValues = chromosomeToCheck.getObjectiveValues();
        int xValueChromosomeToCheck = objectiveValues.get(0);
        int highestXValueInCurrentParetoFront = paretoChromosomes.firstKey();
        Chromosome dominatedChromosome = paretoChromosomes.get(highestXValueInCurrentParetoFront);
        return isDominatingInAllDimensions;
    }
}
