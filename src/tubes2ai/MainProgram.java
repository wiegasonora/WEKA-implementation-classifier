/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;
import java.util.Scanner;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 *
 * @author user
 */
public class MainProgram {
    private static final String filename = "C:\\\\Users\\\\user\\\\Documents\\\\NetBeansProjects\\\\TubesAIWeka\\\\test\\\\iris.arff";
    private static int lastIndex;
    private static FFDDClassifier S;

    public static void main(String Args[]) throws Exception {
        Instances I;
        I = loadArff();
        
    }
    
    
    public static Instances loadArff() throws Exception {
        DataSource source = new DataSource(filename);
        Instances train = source.getDataSet();
        lastIndex = train.numAttributes() - 1;
        if (train.classIndex() == -1)
          train.setClassIndex(lastIndex);
        return train;
    }
    
    public static void convertToNumeric() {
        
    }
    
    public static void classifyNewInstance(Instances train) throws Exception {        
        double index = S.classifyInstance(train.lastInstance());
        String className = train.attribute(lastIndex).value((int)index);
        train.instance(train.numInstances()-1).setClassValue(className);
    }
}

