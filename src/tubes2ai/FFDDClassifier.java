/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author user
 */
public class FFDDClassifier implements Classifier {
    private static int numAttributes;
    private static Instances mainInstances;
    
    private double[][] inWeight;
    private double[][] outWeight;
    private double[] output;

    
    private int numInput; //number of attributes in the Instances
    private int numOutput; //number of classes in the Instances
    private int numClasses;
    private int numHiddenNeuron = 0;
    private double learningRate;
    private boolean isMulti = false;
    
    private double[] dataInput;
    private double[][] dataTraining;
    
    @Override
    public void buildClassifier(Instances input) throws Exception {
        int zz=0;
        int i ;
        int idxArrClass;
        boolean isInClass;
        mainInstances = new Instances(input);
        double[] arrClass = new double[mainInstances.numClasses()];
        for (idxArrClass = 0; idxArrClass < arrClass.length; idxArrClass++){
            arrClass[idxArrClass] = 0;
        }
        numAttributes = input.numAttributes();
        int totInst = mainInstances.numInstances();
        int n = mainInstances.numAttributes();
        dataTraining = new double[totInst][n];
        int idxClass = mainInstances.classIndex();
        Enumeration enu = mainInstances.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance currInstance = (Instance) enu.nextElement();
            isInClass= false;
            for (i = 0; i < n; i++)
            {
                dataTraining[zz][i] = currInstance.value(i);
            }
            int zzz = 0;
            int xxx = 0;
            while (zzz < arrClass.length || isInClass){
                if (arrClass[zzz] == dataTraining[zz][i]){
                    isInClass = true;
                }
                if (arrClass[zzz] != 0){
                    xxx++;
                }
                zzz++;
            }
            if (!isInClass){
               arrClass[xxx] = dataTraining[zz][i];
            }
            zz++;
        }
        
        //ganti jadi id
        for (i = 0; i < totInst; i++){
            int idxtemp = 0;
            while (idxtemp < arrClass.length){
                if (arrClass[idxtemp] != dataTraining[i][idxClass]){
                    idxtemp++;
                }
            }
            dataTraining[i][idxClass] = idxtemp;
        }
        
        //geser
        double[] arrSwapTemp = new double[mainInstances.numInstances()];
        for (i = 0; i < totInst; i++){
            arrSwapTemp[i] = dataTraining[i][idxClass];
        }
        
        for (i = 0; i < totInst; i++){
            for (int j = idxClass; j < numAttributes; j++){
                dataTraining[i][j] = dataTraining[i][j+1];
                if (j == numAttributes -1 ){
                    dataTraining[i][j] = arrSwapTemp[i];
                }
            }
        }
        
        
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double learningRateInput = 0.5;
        //MyNeuralModel neural = new MyNeuralModel(dataTraining, numClasses, numAttributes, numHiddenNeuron, learningRateInput);
        double[] arrOutputClassifiy = new double[instnc.numClasses()];
        //harusnya calcSigmoid return array
        //neural.calcSigmoid(numInput, dataInput, 0);
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

   
}
