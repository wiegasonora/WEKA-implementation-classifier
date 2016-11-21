/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author user
 */
public class Tubes2AI {

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
        DataSource data = new DataSource("mush.arff");
        Instances data1 = data.getDataSet();
        data1.setClassIndex(0);
        numInput = data1.numAttributes();
        numOutput = data1.numClasses();
        numHiddenNeuron = 10;
        isMulti = true;
        
        
        FFDDClassifier FFNN;
        FFNN = new FFDDClassifier();
        FFNN.buildClassifier(data1);
        Normalize norm = new Normalize();
        norm.setInputFormat(data1);
        Filter.useFilter(data1, norm);
        
        Evaluation eval = new Evaluation(data1);
        eval.evaluateModel(FFNN, data1);
        System.out.println(eval.toMatrixString("=====Confusion matrix====="));
        System.out.println(eval.toSummaryString("\nResult:",true));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("fMeasure = " + eval.fMeasure(1));
        System.out.println("Error rate = " + eval.errorRate());  
        
        
    }
    
}
