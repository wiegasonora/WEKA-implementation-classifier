/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import java.util.Scanner;
import java.io.Serializable;

/**
 *
 * @author user
 */
public class Tubes2AI implements Serializable {
    
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        double[][] inWeight;
        double[][] outWeight;
        double[] errOut;
        double[] outputVal;
        double[] hiddenVal;

        int numInput; //number of attributes in the Instances
        int numOutput; //number of classes in the Instances
        int numHiddenNeuron;
        double learningRate;
        double bias;
        Scanner in = new Scanner(System.in);
        boolean isMulti = false;
        
        //load dataset from input file
        
        System.out.print("Masukkan nama file: ");
        String inputFile = in.next();
        DataSource data = new DataSource(inputFile);
        Instances data1 = data.getDataSet();
        
        //set class index
        System.out.print("Masukkan index class: ");
        int inputIdxClass = in.nextInt();
        data1.setClassIndex(inputIdxClass);
        numInput = data1.numAttributes();
        numOutput = data1.numClasses();
        isMulti = true;
        
        
        Classifier FFNN;
        FFNN = new FFDDClassifier();
        System.out.println("1. Load model");
        System.out.println("2. Build new");
        System.out.print("Masukkan pilihan : ");
        int inputOptionModel = in.nextInt();
        
        if (inputOptionModel == 1){
            System.out.print("Masukkan nama file model : ");
            String inputFileModel = in.next();
            FFNN = (FFDDClassifier) weka.core.SerializationHelper.read(inputFileModel);
            
        } else if (inputOptionModel == 2){
            FFNN.buildClassifier(data1);
        }
        
        Normalize norm = new Normalize();
        norm.setInputFormat(data1);
        Filter.useFilter(data1, norm);
        
        DataSource data21 = new DataSource("Team_test.arff");
        Instances data12 = data21.getDataSet();
        data12.setClassIndex(12);
        Evaluation eval = new Evaluation(data12);
        System.out.println("1. Full Training");
        System.out.println("2. 10-Fold Cross Validation");
        System.out.print("Pilih model evaluasi:");
        int inputOption = in.nextInt();
        if (inputOption == 1){
            eval.evaluateModel(FFNN, data1);
            weka.core.SerializationHelper.write("FFNN.model", FFNN);
        } else if (inputOption == 2){
            eval.crossValidateModel(FFNN, data1, 4, new Random(1));
            
            weka.core.SerializationHelper.write("FFNN.model", FFNN);
        }
        
        
        System.out.println(eval.toMatrixString("=====Confusion matrix====="));
        System.out.println(eval.toSummaryString("\nResult:",true));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("fMeasure = " + eval.fMeasure(1));
        System.out.println("Error rate = " + eval.errorRate());  
        
        
    }
    
}
