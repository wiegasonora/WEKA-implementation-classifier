/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import static java.lang.Math.exp;
import java.text.DecimalFormat;
import java.util.Arrays;
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
        double[][] A = new double[3][3];
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (j==2) {
                    A[i][j] = (double)i%2;
                   
                }
                else { 
                     A[i][j] = i+1+j;
                }
            }
        }
        System.out.println(Arrays.deepToString(A));
        MyNeuralModel MLP = new MyNeuralModel(2,2,1,0.5,1); //numClasses,numAttribute,numHidden,learning rate,bias;
        
        double[][] in = new double[3][2];
        in[0][1] = 0.5;
        in[1][1] = 0.1;
        in[2][1] = 0.8;
        
        double[][] out = new double[2][2];
        out[0][0] = 0.5;
        out[0][1] = 0.5;
        out[1][0] = 0.3;
        out[1][1] = 0.9;        
        MLP.customWeight(in, out);
        
        //MLP.initializeWeight();
        
        if (MLP.isMulti()) {
            MLP.printMulti();
        }
        else {
            MLP.printSingle();
        }
        
        int w = 1;
        double d[][] = new double[1][2];
        d[0][0] = 0.35;
        d[0][1] = 0.9;
        MLP.backPropagation(A, 3);
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
    
    public MyNeuralModel(int numClasses, int numAttributes, int hiddenNeurons, double learningRateInput, double b) {
        
        numInput = numAttributes;
        numOutput = numClasses;
        bias = b;
        learningRate = learningRateInput;
        
        if (hiddenNeurons >0) {
            isMulti = true;
            numHiddenNeuron = hiddenNeurons;
            hiddenVal = new double[numHiddenNeuron+1];
        }
        outputVal = new double[numOutput];
        
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
            outWeight = new double[numHiddenNeuron+1][numOutput];            
            for (int i = 0; i <= numHiddenNeuron; i++) {
                for (int j = 0; j <= numOutput; j++) {
                    outWeight[i][j] = generateWeight();
                }
            }
        }
        else {
            //outWeight -> the weight between input nodes and output nodes
            outWeight = new double[numInput+1][numOutput];            
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
                for (int j = 0; j < numOutput; j++) {
                    //updating weight value
                    //System.out.print(errOutput[j]+" ");
                    System.out.println(outWeight[i][j]);
                    outWeight[i][j] += outputVal[j]*errOutput[j]*learningRate;
                    System.out.println(outWeight[i][j]);
                }
            }
        } else {
            for (int i = 0; i <= numInput; i++) {
                for (int j = 1; j <= numHiddenNeuron; j++) {
                    if (i ==0) {
                        inWeight[i][j] += bias*errHidden[j]*learningRate;
                    }
                    else {
                        System.out.println("punya in"+i+" "+j);
                        System.out.println("awal "+inWeight[i][j]);
                        System.out.println("eror nya"+errHidden[j]+" "+"input "+dataInput[i]);
                        inWeight[i][j] += dataInput[i]*errHidden[j]*learningRate;
                        System.out.println("akhir "+inWeight[i][j]);
                    }
                }
            }
            
            for (int i = 0; i <= numHiddenNeuron; i++) {
                for (int j = 0; j < numOutput; j++) {
                    if (i==0) {
                        System.out.println("punya out"+i+" "+j);
                        System.out.println("awal "+outWeight[i][j]);
                        System.out.println("eror nya"+errOutput[j]+" "+"hiddennya "+bias);
                        
                        outWeight[i][j] += bias*errOutput[j]*learningRate;
                        System.out.println("w bias="+outWeight[i][j]);
                    }
                    else {
                        System.out.println("punya out"+i+" "+j);
                        System.out.println("awal "+outWeight[i][j]);
                        System.out.println("eror nya"+errOutput[j]+" "+"hiddennya "+hiddenVal[i]);
                        outWeight[i][j] += hiddenVal[i]*errOutput[j]*learningRate;
                        System.out.println(outWeight[i][j]);
                    }
                }
            }
        }
    }
    
    public void backPropagation(double[][] example,int numEx) {
        double[] errHid = new double[numHiddenNeuron+1];
        double[] currData = new double[numInput+2]; 
        double target, targetVal;
        
        double threshold = 0.5;
        double sumError = 10;
        double[] instError = new double[numEx];
        int k;
        errOut = new double[numOutput];
        while (sumError>threshold) {
            for (int i = 0; i < numEx; i++) {
                for (k = 0; k <= numInput;k++) {
                    currData[k+1] = example[i][k];
                    //System.out.println(currData[k+1]);
                }
                targetVal = example[i][k-1];
                System.out.println("target: "+targetVal);
                
                System.out.println("before update: "+Arrays.toString(currData));
                if (!isMulti) {
                    //single
                    for (int j = 1; j <= numOutput; j++) {
                        System.out.println("class: "+getClassFromData(currData));
                        if (j==(1+targetVal)) {
                            target = 1;
                        }
                        else {
                            target = 0;
                        }
                        
                        errOut[j] = calcOutputError(j,currData,target);
                    }
                    updateWeight(currData,errOut,null);
                }
                else { //multi
                    
                    //calculate output Error
                    for (int l = 1; l<=numHiddenNeuron; l++) {
                        hiddenVal[l] = calcSigmoid(l,currData,'h');
                    }
                    System.out.println("hidden output: "+Arrays.toString(hiddenVal));
                    for (int j = 0; j < numOutput; j++) {
                        System.out.println("class: "+getClassFromData(currData));
                        if (j==(targetVal)) {
                            target = 1;
                        }
                        else {
                            target = 0;
                        }
                        System.out.println("targVal: "+target);
                        errOut[j] = calcOutputError(j,hiddenVal,target);
                    }
                    System.out.println("Output ERROR: "+Arrays.toString(errOut));
                    for (int j = 0; j <= numHiddenNeuron; j++) {
                        errHid[j] = calcHiddenError(j,currData);
                    }
                    System.out.println("Hidden ERROR: "+Arrays.toString(errHid));
                    
                    
                    
                    System.out.println(Arrays.toString(currData));
                    updateWeight(currData,errOut,errHid);
                }
                printMulti();
                instError[i] = 0;
                sumError = 0;
            }
            
        }
    }
    
    
    public double calcHiddenError(int nodeId, double[] dataInput) {
        double errorRate;
        double sigma = calcSigmoid(nodeId,dataInput,'h');
        
        double outSigma = 0;
        for (int i=0; i<numOutput;i++) {
            outSigma += errOut[i]*outWeight[nodeId][i];
        }
        errorRate = sigma*(1-sigma)*(outSigma);
        return errorRate;
    }
    
    public double calcOutputError(int nodeId, double[] dataInput, double target) {
        System.out.println("275 "+Arrays.toString(dataInput));
        double errorRate;
        char a;
        if (isMulti) {
            a = 'o';
        } else {
            a = 's';
        }
        
        double sigma = calcSigmoid(nodeId,dataInput,a);
        System.out.println(Arrays.toString(outputVal));
        outputVal[nodeId] = sigma;
        System.out.println("292 output ="+ sigma);
        errorRate = sigma*(1-sigma)*(target-sigma);
        System.out.println("294 error ="+errorRate);
        return errorRate;
    }
    
    public double calcSigmoid(int nodeId, double[] dataInput, char status) {
        System.out.println("283 "+Arrays.toString(dataInput));
        double val = 0;
        int i;
        switch (status) {
            case 'o': {
                System.out.println(numHiddenNeuron);
                for (i = 1; i <= numHiddenNeuron ; i++) {
                    System.out.println("multi out w="+outWeight[i][nodeId]+" at="+dataInput[i]);
                    val += dataInput[i]*outWeight[i][nodeId];
                }   
                System.out.println("multi out wbias="+outWeight[0][nodeId]+" bias="+bias);
                System.out.println("310 val "+val);
                val+=bias*outWeight[0][nodeId];
                break;
                }
            case 'h': {
                for (i = 1; i <= numInput ; i++) {
                    System.out.println("hidden w="+inWeight[i][nodeId]+" at="+dataInput[i]);
                    val += dataInput[i]*inWeight[i][nodeId];
                }   
                val+=bias*inWeight[0][nodeId];
                break;
                }
            case 's': {
                for (i = 1; i <= numInput ; i++) {
                    System.out.println("single out w="+outWeight[i][nodeId]+" at="+dataInput[i]);
                    val += dataInput[i]*outWeight[i][nodeId];
                }   
                val+=bias*outWeight[0][nodeId];
                break;
                }
            default:
                break;
        }
        
        System.out.println("329 "+val);
        val = 1 / (1+exp((-1)*val));
        System.out.println("331 "+val);
        
        
        return val;
    }
    
    
    public double[] classifyInstOri(double[] dataInput) {
        int o = getNumOutput();
        int h = getNumHiddenNeuron();
        
        double[] outVal = new double[o+1];
        if (!isMulti) {
            for (int i = 1; i<=o; i++) {
                outVal[i] = calcSigmoid(i,dataInput,'s'); 
            }
        }
        else {
            double[] hidVal = new double[h+1];
            for (int i = 1; i<=h; i++) {
                hidVal[i] = calcSigmoid(i, dataInput, 'h');
            }
            for (int i = 1; i<=o; i++) {
                outVal[i] = calcSigmoid(i, hidVal, 'o');
            }
        }
        
        return outVal;
    }
    
    public void printSingle() {
        DecimalFormat df = new DecimalFormat("#.#####");

        for (int i = 0; i <= getNumInput(); i++) {
            for (int j = 1; j <= getNumOutput(); j++) {
                System.out.printf(df.format(outWeight[i][j])+" ");
            }
            System.out.println("");
        }
    }
    
    public void printMulti() {
        DecimalFormat df = new DecimalFormat("#.#####");

        for (int i = 0; i <= getNumInput(); i++) {
            for (int j = 0; j <= getNumHiddenNeuron(); j++) {
                if (i==0||j==0) {
                    System.out.print("(bias)");
                }
                System.out.print(df.format(inWeight[i][j])+" ");
            }
            System.out.println("");
        }

        System.out.println("-===--------===-");
        for (int i = 0; i <= getNumHiddenNeuron(); i++) {
            for (int j = 0; j < getNumOutput(); j++) {
                if (i==0) {
                    System.out.print("(bias)");
                }
                System.out.printf(df.format(outWeight[i][j])+" ");
            }
            System.out.println("");
        }
    }
    
    public void customWeight(double[][] in, double[][] out) {
        if (!isMulti) {
            outWeight = out;            
        }
        else {
            inWeight = in;
            outWeight = out;
        }
    }
    
    public static double getClassFromData(double[] data) {
        double last;
        
        last = data[data.length-1];
        return last;
    }
}
