/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

/**
 *
 * @author user
 */
public class OurNeuron {
    private double[][] weight;
    private int numInput;
    private int numOutput;
    private int numHiddenNeuron;
    private static double bias;
    private final double learningRate;
    
    private double[] dataInput;
    
    public OurNeuron(double[] inputData, boolean isAnyHidden, int hiddenNeurons, double learningRateInput) {
        
        numInput = inputData.length;
        numHiddenNeuron = hiddenNeurons;
        dataInput = inputData;
        learningRate = learningRateInput;
        
        if (isAnyHidden) {
            
        }
        else {
            
        }
    }
    
    public void initializeWeight() {
        
    }
    
    public void updateWeight() {
        
    }
    
    public void backPropagation() {
        
    }
    
    public double calcHiddenError() {
        double total = 0;
       
        
        return total;
    }
    
    public double calcOutputError() {
        double total = 0;
        
        return total;
    }
    
}
