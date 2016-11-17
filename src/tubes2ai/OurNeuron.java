/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import java.util.Random;

/**
 *
 * @author user
 */
public class OurNeuron {
    private double[][] inWeight;
    private double[][] outWeight;
    private double[] output;
    private int numInput; //number of attribute in the Instances
    private int numOutput; //number of class in the Instances
    private int numHiddenNeuron = 0;
    private final double learningRate;
    private boolean isMulti = false;
    
    private double[] dataInput;
    
    public OurNeuron(double[][] inputData, int numClasses, int numAttributes, int hiddenNeurons, double learningRateInput) {
        
        numInput = numAttributes;
        numOutput = numClasses;
        
        learningRate = learningRateInput;
        
        if (hiddenNeurons >0) {
            isMulti = true;
            numHiddenNeuron = hiddenNeurons;
        }
        else {
            
        }
    }
    
    public void initializeWeight() {
        if (isMulti) {
            //
            inWeight = new double[numInput+1][numHiddenNeuron+1];            
            for (int i = 1; i <= numInput; i++) {
                for (int j = 1; j <= numHiddenNeuron; j++) {
                    inWeight[i][j] = generateWeight();
                }
            }
            
            outWeight = new double[numOutput+1][numHiddenNeuron+1];            
            for (int i = 1; i <= numOutput; i++) {
                for (int j = 1; j <= numHiddenNeuron; j++) {
                    outWeight[i][j] = generateWeight();
                }
            }
        }
        else {
            outWeight = new double[numOutput+1][numInput+1];            
            for (int i = 1; i <= numOutput; i++) {
                for (int j = 1; j <= numInput; j++) {
                    outWeight[i][j] = generateWeight();
                }
            }
        }
    }
    
    public double generateWeight() {
        double val;
        double rangeMin = -3.0;
        double rangeMax = 3.0;
        
        
        Random r = new Random();
        val = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
        
        return val;
    }
    
    public void updateWeight() {
        
    }
    
    public void backPropagation() {
        
    }
    
    public double calcHiddenError() {
        double total = 0;
        
        
        return total;
    }
    
    public double calcOutputError(int outId) {
        double total = 0;
        
        return total;
    }
    
    public double calcSigmoid(int nodeId) {
        double val = 0;
        if (!isMulti) {
            
        }
        else {
            
        }
        return val;
    }
}
