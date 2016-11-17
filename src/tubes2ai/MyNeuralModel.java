/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import static java.lang.Math.exp;
import java.text.DecimalFormat;
import java.util.Random;

/**
 *
 * @author user
 */
public class MyNeuralModel {
    private double[][] inWeight;
    private double[][] outWeight;
    private double[] outputClasses;

    
    private final int numInput; //number of attributes in the Instances
    private final int numOutput; //number of classes in the Instances
    private int numHiddenNeuron = 0;
    private final double learningRate;
    private boolean isMulti = false;
    
    private double[] dataInput;
    
    public static void main (String[] args) {
        double[][] A = new double[5][6];
        
        for (int i = 1; i <= 4; i++) {
            for (int j = 1; j <= 5; j++) {
                if (j==5) {
                    A[i][j] = 3;
                }
                else { 
                    A[i][j] = 5;
                }
            }
        }
        MyNeuralModel MLP = new MyNeuralModel(A,2,5,1,0.1);
        MLP.initializeWeight();
        double[][] in = MLP.getInWeight();
        double[][] out = MLP.getOutWeight();
        DecimalFormat df = new DecimalFormat("#.##");
        
        for (int i = 1; i <= MLP.getNumInput(); i++) {
            for (int j = 1; j <= MLP.getNumHiddenNeuron(); j++) {
                System.out.print(df.format(in[i][j])+" ");
            }
            System.out.println("");
        }
            
        System.out.println("-===--------===-");
        for (int i = 1; i <= MLP.getNumOutput(); i++) {
            for (int j = 1; j <= MLP.getNumHiddenNeuron(); j++) {
                System.out.printf(df.format(out[i][j])+" ");
            }
            System.out.println("");
        }
        
    }
    
    //Class getter
    public double[][] getInWeight() {
        return inWeight;
    }

    public double[][] getOutWeight() {
        return outWeight;
    }

    public double[] getOutputClasses() {
        return outputClasses;
    }

    public int getNumInput() {
        return numInput;
    }

    public int getNumOutput() {
        return numOutput;
    }

    public int getNumHiddenNeuron() {
        return numHiddenNeuron;
    }

    public boolean isIsMulti() {
        return isMulti;
    }
    
    public MyNeuralModel(double[][] trainData, int numClasses, int numAttributes, int hiddenNeurons, double learningRateInput) {
        
        numInput = numAttributes;
        numOutput = numClasses;
        
        learningRate = learningRateInput;
        
        if (hiddenNeurons >0) {
            isMulti = true;
            numHiddenNeuron = hiddenNeurons;
        }
    }
    
    public void initializeWeight() {
        if (isMulti) {
            //inWeight -> the weight between input nodes and hidden layer nodes
            inWeight = new double[numInput+1][numHiddenNeuron+1];            
            for (int i = 1; i <= numInput; i++) {
                for (int j = 1; j <= numHiddenNeuron; j++) {
                    inWeight[i][j] = generateWeight();
                }
            }
            
            //outWeight -> the weight between hidden layer nodes and output nodes
            outWeight = new double[numOutput+1][numHiddenNeuron+1];            
            for (int i = 1; i <= numOutput; i++) {
                for (int j = 1; j <= numHiddenNeuron; j++) {
                    outWeight[i][j] = generateWeight();
                }
            }
        }
        else {
            //outWeight -> the weight between input nodes and output nodes
            outWeight = new double[numInput+1][numOutput+1];            
            for (int i = 1; i <= numInput; i++) {
                for (int j = 1; j <= numOutput; j++) {
                    outWeight[i][j] = generateWeight();
                }
            }
        }
    }
    
    public double generateWeight() {
        double val;
        double rangeMin = -2.0;
        double rangeMax = 2.0;
        
        Random r = new Random();
        val = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
        
        
        return val;
    }
    
    public void updateWeight() {
        
    }
    
    public void backPropagation(double[][] example,int numEx) {
        double[] errOut = new double[numOutput+1];
        double[] errHid = new double[numHiddenNeuron+1];
        double[] dataInput = new double[numInput]; 
        double target = 0;
        for (int i = 0; i < numEx; i++) {
            
            if (!isMulti) {
                //single
                for (int j = 1; j <= numOutput; j++) {
                    if (j==numOutput) {
                        target = outWeight[j][numOutput];
                    }
                    errOut[j] = calcOutputError(j,dataInput,target);
                }
                updateWeight();
            }
            else {
                //multi
                for (int j = 1; j <= numHiddenNeuron; j++) {
                    errHid[j] = calcHiddenError(j,dataInput);
                }
                
                for (int j = 1; j <= numOutput; j++) {
                    if (j==numOutput) {
                        target = outWeight[j][numOutput];
                    }
                    errOut[j] = calcOutputError(j,dataInput,target);
                }
                
                updateWeight();
            }
        }
        
    }
    
    public double calcHiddenError(int nodeId, double[] dataInput) {
        double errorRate = 0;
        errorRate = calcSigmoid(nodeId,dataInput,'h');
        
        return errorRate;
    }
    
    public double calcOutputError(int nodeId, double[] dataInput, double target) {
        double errorRate = 0;
        double sigma = calcSigmoid(nodeId,dataInput,'o');
        errorRate = sigma*(1-sigma)*(target-sigma);
        return errorRate;
    }
    
    public double calcSigmoid(int nodeId, double[] dataInput, char status) {
        double val = 0;
        int i;
        if (status=='o') {
            for (i = 1; i < dataInput.length ; i++) {
                val += dataInput[i]*outWeight[i][nodeId];
            }
        }
        else if (status=='h'){
            for (i = 1; i < dataInput.length ; i++) {
                val += dataInput[i]*inWeight[i][nodeId];
            }
        }
        
        val = 1 / (1+exp(-val));
        
        return val;
    }
}
