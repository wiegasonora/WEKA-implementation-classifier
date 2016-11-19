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
    public static double[][] inWeight;
    public static double[][] outWeight;
    private double[] errOut;
    private double[] outputVal;
    private double[] hiddenVal;
    
    private final int numInput; //number of attributes in the Instances
    private final int numOutput; //number of classes in the Instances
    private int numHiddenNeuron = 0;
    private final double learningRate;
    private final double bias;

    private boolean isMulti = false;
    
    
    public static void main (String[] args) {
        double[][] A = new double[5][6];
        
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 6; j++) {
                if (j==5) {
                    A[i][j] = i+1;
                }
                else { 
                    A[i][j] = j;
                }
            }
        }
        MyNeuralModel MLP = new MyNeuralModel(A,4,6,2,0.1,1);
        double[][] in = MLP.getInWeight();
        double[][] out = MLP.getOutWeight();
        
        if (MLP.isMulti()) {
            MLP.printMulti();
        }
        else {
            MLP.printSingle();
        }
        
        int w = 100;
        MLP.backPropagation(A, 5,w);
        System.out.println("After BP");
        
        if (MLP.isMulti()) {
            MLP.printMulti();
        }
        else {
            MLP.printSingle();
        }
        
    }
    
    //Class getter
    public double[][] getInWeight() {
        return inWeight;
    }

    public double[][] getOutWeight() {
        return outWeight;
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

    public boolean isMulti() {
        return isMulti;
    }
    
    public MyNeuralModel(double[][] trainData, int numClasses, int numAttributes, int hiddenNeurons, double learningRateInput, double b) {
        
        numInput = numAttributes;
        numOutput = numClasses;
        bias = b;
        learningRate = learningRateInput;
        
        if (hiddenNeurons >0) {
            isMulti = true;
            numHiddenNeuron = hiddenNeurons;
        }
        
        initializeWeight();
    }
    
    public void initializeWeight() {
        if (isMulti) {
            //inWeight -> the weight between input nodes and hidden layer nodes
            inWeight = new double[numInput+1][numHiddenNeuron+1];            
            for (int i = 0; i <= numInput; i++) {
                for (int j = 0; j <= numHiddenNeuron; j++) {
                    inWeight[i][j] = generateWeight();
                }
            }
            
            //outWeight -> the weight between hidden layer nodes and output nodes
            outWeight = new double[numHiddenNeuron+1][numOutput+1];            
            for (int i = 0; i <= numHiddenNeuron; i++) {
                for (int j = 0; j <= numOutput; j++) {
                    outWeight[i][j] = generateWeight();
                }
            }
        }
        else {
            //outWeight -> the weight between input nodes and output nodes
            outWeight = new double[numInput+1][numOutput+1];            
            for (int i = 0; i <= numInput; i++) {
                for (int j = 0; j <= numOutput; j++) {
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
        
        val = val*10000;
        val = Math.round(val);
        val = val /10000;
        return val;
    }
    
    public void updateWeight(double[] dataInput, double[] errOutput, double[] errHidden) {
        if (!isMulti) {
            for (int i = 0; i <= numInput; i++) {
                for (int j = 1; j <= numOutput; j++) {
                    //updating weight value
                    //System.out.println(outWeight[i][j]);
                    outWeight[i][j] += calcSigmoid(j,dataInput,'s')*errOutput[j]*learningRate;
                    //System.out.println(outWeight[i][j]);
                }
            }
        } else {
            for (int i = 0; i <= numInput; i++) {
                for (int j = 0; j <= numHiddenNeuron; j++) {
                    //System.out.println(inWeight[i][j]);
                    
                    inWeight[i][j] += calcSigmoid(j,dataInput,'h')*errHidden[j]*learningRate;
                    //System.out.println(inWeight[i][j]);
                    
                }
            }
            
            for (int i = 0; i <= numHiddenNeuron; i++) {
                for (int j = 1; j <= numOutput; j++) {
                    //System.out.println(outWeight[i][j]);
                    
                    outWeight[i][j] = calcSigmoid(j,dataInput,'o')*errOutput[j]*learningRate;
                    //System.out.println(outWeight[i][j]);
                    
                }
            }
        }
    }
    
    public void backPropagation(double[][] example,int numEx, int numIter) {
        double[] errHid = new double[numHiddenNeuron+1];
        double[] currData = new double[numInput+1]; 
        double target, targetVal;
        hiddenVal = new double[numHiddenNeuron+1];
        int k;
        errOut = new double[numOutput+1];
        for (int a=0; a<=numIter;a++) {
            for (int i = 0; i < numEx; i++) {
                for (k = 0; k < numInput;k++) {
                    currData[k+1] = example[i][k];
                    //System.out.println(currData[k+1]);
                }
                targetVal = example[i][k-1];
                target = 0;

                if (!isMulti) {
                    //single
                    for (int j = 1; j <= numOutput; j++) {
                        if (currData[numOutput]==targetVal) {
                            target = 1;
                        }
                        else {
                            target = 0;
                        }
                        
                        errOut[j] = calcOutputError(j,currData,target);
                    }
                    updateWeight(currData,errOut,null);
                }
                else {
                    //multi
                    for (int j = 1; j <= numHiddenNeuron; j++) {
                        errHid[j] = calcHiddenError(j,currData);
                    }
                    
                    for (int l = 1; l<=numHiddenNeuron; l++) {
                        hiddenVal[l] = calcSigmoid(l,currData,'h');
                    }
                    
                    for (int j = 1; j <= numOutput; j++) {
                        if (currData[numOutput]==targetVal) {
                            target = 1;
                        }
                        else {
                            target = 0;
                        }
                        errOut[j] = calcOutputError(j,currData,target);
                    }

                    updateWeight(currData,errOut,errHid);
                }
            }
        }
    }
    
    
    /*
        MASIH EROOR
    */
    public double calcHiddenError(int nodeId, double[] dataInput) {
        double errorRate;
        double sigma = calcSigmoid(nodeId,dataInput,'h');
        
        double outSigma = 0;
        for (int i=1; i<=numOutput;i++) {
            outSigma += errOut[i]*outWeight[nodeId][i];
        }
        errorRate = sigma*(1-sigma)*(outSigma);
        return errorRate;
    }
    
    public double calcOutputError(int nodeId, double[] dataInput, double target) {
        double errorRate;
        double sigma = calcSigmoid(nodeId,dataInput,'o');
        errorRate = sigma*(1-sigma)*(target-sigma);
        return errorRate;
    }
    
    public double calcSigmoid(int nodeId, double[] dataInput, char status) {
        double val = 0;
        int i;
        switch (status) {
            case 'o': {
                for (i = 1; i <= numHiddenNeuron ; i++) {
                    //System.out.println("i="+i+" w="+outWeight[i][nodeId]+" at="+dataInput[i+1]);
                    val += dataInput[i]*outWeight[i][nodeId];
                }   
                val+=bias*outWeight[0][nodeId];
                break;
                }
            case 'h': {
                for (i = 1; i <= numInput ; i++) {
                    //System.out.println("i="+i+" w="+inWeight[i][nodeId]+" at="+dataInput[i+1]);
                    val += dataInput[i]*inWeight[i][nodeId];
                }   
                val+=bias*inWeight[0][nodeId];
                break;
                }
            case 's': {
                for (i = 1; i <= numInput ; i++) {
                    //System.out.println("i="+i+" w="+outWeight[i][nodeId]+" at="+dataInput[i+1]);
                    val += dataInput[i]*outWeight[i][nodeId];
                }   
                val+=bias*outWeight[0][nodeId];
                break;
                }
            default:
                break;
        }
        
        
        val = 1 / (1+exp((-1)*val));
        
        return val;
    }
    
    
    public static double[] classifyInstOri(MyNeuralModel M, double[] dataInput) {
        int o = M.getNumOutput();
        int h = M.getNumHiddenNeuron();
        
        double[] outVal = new double[o+1];
        if (!M.isMulti()) {
            for (int i = 1; i<=o; i++) {
                outVal[i] = M.calcSigmoid(i,dataInput,'s'); 
            }
        }
        else {
            double[] hidVal = new double[h+1];
            for (int i = 1; i<=h; i++) {
                hidVal[i] = M.calcSigmoid(i, dataInput, 'h');
            }
            for (int i = 1; i<=o; i++) {
                outVal[i] = M.calcSigmoid(i, hidVal, 'o');
            }
        }
        
        return outVal;
    }
    
    public void printSingle() {
        DecimalFormat df = new DecimalFormat("#.#####");

        for (int i = 1; i <= getNumInput(); i++) {
            for (int j = 1; j <= getNumOutput(); j++) {
                System.out.printf(df.format(outWeight[i][j])+" ");
            }
            System.out.println("");
        }
    }
    
    public void printMulti() {
        DecimalFormat df = new DecimalFormat("#.#####");

        for (int i = 1; i <= getNumInput(); i++) {
            for (int j = 1; j <= getNumHiddenNeuron(); j++) {
                System.out.print(df.format(inWeight[i][j])+" ");
            }
            System.out.println("");
        }

        System.out.println("-===--------===-");
        for (int i = 1; i <= getNumHiddenNeuron(); i++) {
            for (int j = 1; j <= getNumOutput(); j++) {
                System.out.printf(df.format(outWeight[i][j])+" ");
            }
            System.out.println("");
        }
    }
}
