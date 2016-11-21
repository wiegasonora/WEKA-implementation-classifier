/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import java.util.Arrays;
import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
/**
 *
 * @author user
 */
public class FFDDClassifier extends AbstractClassifier {
    private static int numAttributes;
    private static Instances mainInstances;
    
    private double[][] inWeight;
    private double[][] outWeight;
    private double[] output;
    private double[] errOut;
    private double[] outputVal;
    private double[] hiddenVal;

    
    private int numInput; //number of attributes in the Instances
    private int numOutput; //number of classes in the Instances
    private int numClasses;
    private int numHiddenNeuron = 10;
    private double learningRate;
    private boolean isMulti = false;
    private MyNeuralModel MLP;
    private double[] dataInput;
    private double[][] dataTraining;
    
    @Override
    public void buildClassifier(Instances input) throws Exception {
        
        Normalize norm = new Normalize();
        norm.setInputFormat(input);
        Filter filter = new NominalToBinary();
        Filter.useFilter(input, norm);
        
        int zz=0;
        int i ;
        int idxArrClass;
        boolean isInClass;
        
        mainInstances = new Instances(input);
       
        numAttributes = input.numAttributes()-1;
        int totInst = mainInstances.numInstances();
        int n = mainInstances.numAttributes();
        dataTraining = new double[totInst][n];
        int idxClass = mainInstances.classIndex();
        System.out.println(idxClass);
        Enumeration enu = mainInstances.enumerateInstances();
        
        //fill the dataTraining
        while (enu.hasMoreElements()) {
            
            Instance currInstance = (Instance) enu.nextElement();
            for (i = 0; i < n; i++)
            {
                if (i == n){
                    dataTraining[zz][i] = currInstance.classValue(); 
                } else {
                    dataTraining[zz][i] = currInstance.value(i);   
                }
            }
            zz++;
            
        }
        
        
        System.out.println(Arrays.deepToString(dataTraining));
        if (idxClass != numAttributes-1){
            //geser
            //fill arrSwapTemp with class from all instances
            double[] arrSwapTemp = new double[mainInstances.numInstances()];
            for (i = 0; i < totInst; i++){
                arrSwapTemp[i] = dataTraining[i][idxClass];
            }

            int j = idxClass;

            //overwrite arrSwapTemp to most right attribute
            for (i = 0; i < totInst; i++){
                while (j <= numAttributes -1){
                    if (j == numAttributes -1 ){
                        dataTraining[i][j] = arrSwapTemp[i];
                    } else {
                        dataTraining[i][j] = dataTraining[i][j+1];
                    }
                    j++;
                }
            }
        }
        
        System.out.println(Arrays.deepToString(dataTraining));
        //System.out.println(Arrays.deepToString(dataTraining));
        //create model
        double learningrate = 0.001;
        numClasses = input.numClasses();
        MLP = new MyNeuralModel(numClasses, numAttributes, numHiddenNeuron, learningrate, 0);
        MLP.initializeWeight();
        MLP.backPropagation(dataTraining, input.numInstances());
        inWeight = MLP.getInWeight();   //koreksi
        outWeight = MLP.getOutWeight(); //koreksi
        
        
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        
        //initialize array
        double[] arrOutputClassify = new double[instnc.numClasses()];
        for (int i = 0; i < arrOutputClassify.length; i++){
            arrOutputClassify[i] = 0;
        }
        
        double[] arrInput = new double[instnc.numAttributes()];
        for (int i = 1; i < instnc.numAttributes(); i++){
            arrInput[i] = instnc.value(i-1);  /* GABISA KALAU KELAS DI TENGAH  */
        }
        //System.out.println(Arrays.toString(arrInput));
        //generate sigmoid
        int o = MLP.getNumOutput();
        int h = MLP.getNumHiddenNeuron();
        
        if (!MLP.isMulti()) {
            for (int i = 0; i<o; i++) {
                arrOutputClassify[i] = MLP.calcSigmoid(i,arrInput,'s'); 
            }
        }
        else {
            double[] hidVal = new double[h+1];
            for (int i = 1; i<=h; i++) {
                hidVal[i] = MLP.calcSigmoid(i, arrInput, 'h');
                //System.out.println(hidVal[i]);
            }
            for (int i = 0; i< o; i++) {
                arrOutputClassify[i] = MLP.calcSigmoid(i, hidVal, 'o');
            }
        }
        
        //find max sigmoid
        int maxSigmoidId = 0;
        System.out.println("hasil = "+Arrays.toString(arrOutputClassify));
        for (int i = 0; i < arrOutputClassify.length ; i++){
            if (arrOutputClassify[i] > arrOutputClassify[maxSigmoidId]){
               maxSigmoidId = i;
            }
        }
        //System.out.println((maxSigmoidId));
        return (double) (maxSigmoidId);
        
        //classifying
        
        
        //pelajarin normalisasi utk 
        
    }

    
}
