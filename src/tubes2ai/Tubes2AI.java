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
public class Tubes2AI  {

    /**
     * @param args the command line arguments
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

        boolean isMulti = false;
        
        //load dataset from iris.arff
        Scanner in = new Scanner(System.in);
        System.out.print("Masukkan nama file: ");
        String inputFile = in.next();
        DataSource data = new DataSource(inputFile);
        Instances data1 = data.getDataSet();
        System.out.print("Masukkan index class: ");
        int inputIdxClass = in.nextInt();
        data1.setClassIndex(inputIdxClass);
        numInput = data1.numAttributes();
        numOutput = data1.numClasses();
        isMulti = true;
        
        
        FFDDClassifier FFNN;
        FFNN = new FFDDClassifier();
        FFNN.buildClassifier(data1);
        Normalize norm = new Normalize();
        norm.setInputFormat(data1);
        Filter.useFilter(data1, norm);
        
        Evaluation eval = new Evaluation(data1);
        System.out.println("1. Full Training");
        System.out.println("2. 10-Fold Cross Validation");
        System.out.print("Pilih model evaluasi:");
        int inputOption = in.nextInt();
        if (inputOption == 1){
            eval.evaluateModel(FFNN, data1);
            //weka.core.SerializationHelper.write("FFNN.model", FFNN);
        } else if (inputOption == 2){
            eval.crossValidateModel(FFNN, data1, 10, new Random(1));
            //weka.core.SerializationHelper.write("FFNN.model", FFNN);
        }
        
        System.out.println(eval.toMatrixString("=====Confusion matrix====="));
        System.out.println(eval.toSummaryString("\nResult:",true));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("fMeasure = " + eval.fMeasure(1));
        System.out.println("Error rate = " + eval.errorRate());  
        
        
    }
    
}
